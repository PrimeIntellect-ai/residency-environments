// file: test_mk_warp_pipeline_mbarrier.cu

#include "mk_warp_pipeline_mbarrier_common.h"
#include "mk_warp_pipeline_mbarrier_oracle.hpp"

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

static constexpr uint64_t g_state = 0x51d9f0c3a7b2e641ULL;
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
    uint64_t clock = 0, event_seq = 0, pipe = 0, buf = 0, bar = 0, tile = 0, async = 0, state = 0;
};

struct Scenario {
    std::string name;
    MkwpProblemSpec spec;
    std::vector<MkwpRunSpec> ops;
};

static MkwpProblemSpec make_spec(int buffers, int lw, int cw, int sw, int max_tiles,
                                 int max_pending, int max_role_queue, int max_steps) {
    MkwpProblemSpec s = {};
    s.abi_version = MKWP_ABI_VERSION;
    s.buffer_count = buffers;
    s.loader_warps = lw; s.compute_warps = cw; s.storer_warps = sw;
    s.barrier_count = buffers * MKWP_BARRIERS_PER_BUFFER;
    s.max_tiles = max_tiles;
    s.max_pending_async = max_pending;
    s.max_role_queue = max_role_queue;
    s.max_steps = max_steps;
    s.flags = 0;
    if (!mkwp_validate_problem_spec(&s)) throw std::runtime_error("invalid MkwpProblemSpec");
    return s;
}

static MkwpRunSpec mk(int op_kind, int step_id) {
    MkwpRunSpec r = {};
    r.abi_version = MKWP_ABI_VERSION;
    r.op_kind = op_kind;
    r.step_id = step_id;
    return r;
}
static MkwpRunSpec op_enq(int step, uint64_t tile, uint64_t lb, uint64_t ci, uint64_t sb, uint64_t seed) {
    MkwpRunSpec r = mk(MKWP_OP_ENQUEUE_TILE, step);
    r.a_tile = tile; r.a_load_bytes = lb; r.a_compute_iters = ci; r.a_store_bytes = sb; r.a_seed = seed;
    return r;
}
static MkwpRunSpec op_load(int step, int loader, int limit) { MkwpRunSpec r = mk(MKWP_OP_LOADER_STEP, step); r.a_role_id = loader; r.a_limit = limit; return r; }
static MkwpRunSpec op_comp(int step, int cid, int limit) { MkwpRunSpec r = mk(MKWP_OP_COMPUTE_STEP, step); r.a_role_id = cid; r.a_limit = limit; return r; }
static MkwpRunSpec op_store(int step, int sid, int limit) { MkwpRunSpec r = mk(MKWP_OP_STORER_STEP, step); r.a_role_id = sid; r.a_limit = limit; return r; }
static MkwpRunSpec op_adv(int step, uint64_t delta, int max_async) { MkwpRunSpec r = mk(MKWP_OP_ADVANCE, step); r.a_delta = delta; r.a_limit = max_async; return r; }
static MkwpRunSpec op_cancel(int step, uint64_t tile) { MkwpRunSpec r = mk(MKWP_OP_CANCEL_TILE, step); r.a_tile = tile; return r; }
static MkwpRunSpec op_reset(int step, int bar) { MkwpRunSpec r = mk(MKWP_OP_RESET_BARRIER, step); r.a_barrier = bar; return r; }

// ------------------------------------------------------------- scenarios
// S1: basic full pipeline through all three roles + TILE_DONE + id reuse.
static Scenario sc_basic_pipeline() {
    Scenario s; s.name = "basic_full_pipeline";
    s.spec = make_spec(2, 1, 1, 1, 32, 64, 64, 256);
    int t = 0;
    s.ops.push_back(op_enq(t++, 1, 5, 7, 3, 0x1111));
    s.ops.push_back(op_load(t++, 0, 1));     // buffer0 LOAD_INFLIGHT, async due clk5
    s.ops.push_back(op_comp(t++, 0, 1));     // COMPUTE_NO_READY (load not done)
    s.ops.push_back(op_adv(t++, 5, 4));      // LOAD_COMPLETE: buffer0 LOAD_READY, compute ready
    s.ops.push_back(op_comp(t++, 0, 1));     // COMPUTE_ISSUE
    s.ops.push_back(op_store(t++, 0, 1));    // STORER_NO_READY (compute not done) -- wait, store queue empty -> NO_READY
    s.ops.push_back(op_adv(t++, 7, 4));      // COMPUTE_COMPLETE -> store ready
    s.ops.push_back(op_store(t++, 0, 1));    // STORE_ISSUE
    s.ops.push_back(op_adv(t++, 3, 4));      // STORE_COMPLETE + TILE_DONE
    // reuse id 1 (now DONE)
    s.ops.push_back(op_enq(t++, 1, 2, 2, 2, 0x2222));
    s.ops.push_back(op_load(t++, 0, 1));
    s.ops.push_back(op_adv(t++, 2, 8));
    s.ops.push_back(op_comp(t++, 0, 1));
    s.ops.push_back(op_adv(t++, 2, 8));
    s.ops.push_back(op_store(t++, 0, 1));
    s.ops.push_back(op_adv(t++, 2, 8));
    return s;
}

// S2: multi-buffer pressure -> LOADER_NO_BUFFER; round-robin slot reuse; batch
//     loader limit; interleaved compute/store draining.
static Scenario sc_multibuffer() {
    Scenario s; s.name = "multibuffer_pressure";
    s.spec = make_spec(3, 1, 1, 1, 64, 128, 64, 512);
    int t = 0;
    for (uint64_t id = 1; id <= 6; ++id) s.ops.push_back(op_enq(t++, id, 4, 4, 4, id * 10));
    s.ops.push_back(op_load(t++, 0, 5));   // only 3 buffers -> 3 LOAD_ISSUE then LOADER_NO_BUFFER
    s.ops.push_back(op_adv(t++, 4, 8));    // all 3 loads complete -> compute ready
    s.ops.push_back(op_comp(t++, 0, 3));   // 3 compute issues
    s.ops.push_back(op_load(t++, 0, 3));   // buffers busy (compute inflight) -> NO_BUFFER
    s.ops.push_back(op_adv(t++, 4, 8));    // 3 compute complete -> store ready
    s.ops.push_back(op_store(t++, 0, 3));  // 3 store issues
    s.ops.push_back(op_adv(t++, 4, 8));    // 3 store complete + tile done -> buffers EMPTY
    s.ops.push_back(op_load(t++, 0, 5));   // now 3 free -> load remaining 3 tiles then NO_BUFFER
    s.ops.push_back(op_adv(t++, 4, 16));
    s.ops.push_back(op_comp(t++, 0, 5));
    s.ops.push_back(op_adv(t++, 4, 16));
    s.ops.push_back(op_store(t++, 0, 5));
    s.ops.push_back(op_adv(t++, 4, 16));
    s.ops.push_back(op_load(t++, 0, 1));   // queue empty -> LOADER_NO_TILE
    return s;
}

// S3: mbarrier phase waits -- compute waits on load, storer waits on compute,
//     with repeated WAIT events (idempotent mask set).
static Scenario sc_mbarrier_waits() {
    Scenario s; s.name = "mbarrier_phase_waits";
    s.spec = make_spec(2, 1, 2, 2, 32, 64, 64, 256);
    int t = 0;
    s.ops.push_back(op_enq(t++, 100, 10, 10, 10, 7));
    s.ops.push_back(op_load(t++, 0, 1));   // load inflight (due clk10)
    s.ops.push_back(op_adv(t++, 3, 4));    // clk3 -> not due, nothing
    // load NOT complete yet but buffer not in compute queue -> COMPUTE_NO_READY
    s.ops.push_back(op_comp(t++, 0, 1));   // NO_READY (compute queue empty)
    s.ops.push_back(op_adv(t++, 7, 4));    // clk10 -> load complete, buffer in compute queue
    s.ops.push_back(op_comp(t++, 1, 1));   // compute id1 -> COMPUTE_ISSUE
    // now storer asks before compute completes: buffer in store queue? no. NO_READY
    s.ops.push_back(op_store(t++, 0, 1));  // STORER_NO_READY
    s.ops.push_back(op_adv(t++, 10, 4));   // compute complete -> store ready
    s.ops.push_back(op_store(t++, 0, 1));  // store issue
    s.ops.push_back(op_adv(t++, 10, 4));   // store complete + done
    // Now explicit WAIT_LOAD: enqueue + load + advance partially, put into compute
    // queue only after load completes. To trigger MBARRIER_WAIT_LOAD we must have a
    // buffer in the compute queue whose load barrier is not done. Cancel won't do.
    // Construct: load tile, advance to complete (in compute queue, load done=1),
    // RESET its load barrier? buffer not empty -> invalid. Instead use a second
    // enqueue where we manually drive compute queue via load-complete then RESET.
    return s;
}

// S4: stale async after cancel of a buffer-owning tile -> ASYNC_STALE_DROP and a
//     replacement tile must NOT be completed by the stale event.
static Scenario sc_stale_cancel() {
    Scenario s; s.name = "stale_async_cancel";
    s.spec = make_spec(2, 1, 1, 1, 32, 64, 64, 256);
    int t = 0;
    s.ops.push_back(op_enq(t++, 1, 20, 5, 5, 0xAA));   // long load
    s.ops.push_back(op_load(t++, 0, 1));               // buffer0, load due clk20, phase1
    s.ops.push_back(op_cancel(t++, 1));                // BUFFER_CANCEL_RELEASE; load barrier phase++ ->2
    // enqueue a new tile, load into the SAME buffer0 (now empty); phase becomes 3.
    s.ops.push_back(op_enq(t++, 2, 4, 5, 5, 0xBB));
    s.ops.push_back(op_load(t++, 0, 1));               // buffer0 reused, load phase=3, due clk4
    s.ops.push_back(op_adv(t++, 20, 8));               // clk20: tile2 load due@4 completes;
                                                       // tile1 stale load due@20 -> STALE_DROP
    s.ops.push_back(op_comp(t++, 0, 1));               // compute tile2
    s.ops.push_back(op_adv(t++, 5, 8));
    s.ops.push_back(op_store(t++, 0, 1));
    s.ops.push_back(op_adv(t++, 5, 8));                // tile2 done
    // Cancel a QUEUED tile -> stays out of pipeline (loader skips it).
    s.ops.push_back(op_enq(t++, 3, 2, 2, 2, 1));
    s.ops.push_back(op_enq(t++, 4, 2, 2, 2, 1));
    s.ops.push_back(op_cancel(t++, 3));                // TILE_CANCEL (queued)
    s.ops.push_back(op_load(t++, 0, 2));               // skips 3, loads 4, then NO_TILE
    s.ops.push_back(op_adv(t++, 2, 8));
    return s;
}

// S5: role-queue ordering by (tile_seq, buffer id) with out-of-order completion.
//     Two buffers complete loads in reverse buffer order but compute must pick the
//     smaller tile_seq first.
static Scenario sc_role_order() {
    Scenario s; s.name = "role_queue_ordering";
    s.spec = make_spec(3, 1, 1, 1, 64, 128, 64, 512);
    int t = 0;
    // tile A (seq1) short load, tile B (seq2) long load, tile C (seq3) medium.
    s.ops.push_back(op_enq(t++, 10, 30, 4, 4, 1));   // seq1 -> buffer0, due30
    s.ops.push_back(op_enq(t++, 20, 5, 4, 4, 2));    // seq2 -> buffer1, due5
    s.ops.push_back(op_enq(t++, 30, 15, 4, 4, 3));   // seq3 -> buffer2, due15
    s.ops.push_back(op_load(t++, 0, 3));             // all three load inflight
    s.ops.push_back(op_adv(t++, 30, 8));             // completes in due order: B(5),C(15),A(30)
    // compute queue now ordered by tile_seq: A(seq1,buf0), B(seq2,buf1), C(seq3,buf2)
    s.ops.push_back(op_comp(t++, 0, 1));             // picks A (smallest seq) buffer0
    s.ops.push_back(op_comp(t++, 0, 1));             // picks B buffer1
    s.ops.push_back(op_comp(t++, 0, 1));             // picks C buffer2
    s.ops.push_back(op_adv(t++, 4, 8));              // all compute complete -> store ready
    s.ops.push_back(op_store(t++, 0, 3));            // store in tile_seq order
    s.ops.push_back(op_adv(t++, 4, 8));
    return s;
}

// S6: RESET_BARRIER valid/invalid + INVALID ops + empty queue ops + reset replay.
static Scenario sc_reset_invalid() {
    Scenario s; s.name = "reset_and_invalid";
    s.spec = make_spec(2, 1, 1, 1, 8, 16, 16, 256);
    int t = 0;
    // RESET_BARRIER on empty buffer's barrier (buffer0 EMPTY) -> valid.
    s.ops.push_back(op_reset(t++, 0));        // load barrier of buffer0
    s.ops.push_back(op_reset(t++, 4));        // compute barrier of buffer1 (id 4)
    s.ops.push_back(op_reset(t++, 99));       // OOB -> invalid
    // Now make buffer0 busy then RESET its barrier -> invalid.
    s.ops.push_back(op_enq(t++, 1, 3, 3, 3, 5));
    s.ops.push_back(op_load(t++, 0, 1));      // buffer0 busy
    s.ops.push_back(op_reset(t++, 0));        // buffer0 non-empty -> invalid
    s.ops.push_back(op_reset(t++, 2));        // barrier 2 = buffer0 store -> invalid (buffer0 busy)
    // invalid step ops
    s.ops.push_back(op_load(t++, 5, 1));      // loader_id oob -> invalid
    s.ops.push_back(op_comp(t++, 0, 0));      // limit 0 -> invalid
    s.ops.push_back(op_store(t++, 3, 1));     // storer_id oob -> invalid
    // enqueue invalid: zero fields, duplicate nonterminal, table full
    s.ops.push_back(op_enq(t++, 2, 0, 1, 1, 1));   // load 0 -> invalid
    s.ops.push_back(op_enq(t++, 1, 1, 1, 1, 1));   // dup nonterminal (id1 loading) -> invalid
    // cancel invalid: absent + terminal
    s.ops.push_back(op_cancel(t++, 999));     // absent -> invalid
    // ADVANCE delta 0 valid (no-op-ish): clk unchanged, processes due (none)
    s.ops.push_back(op_adv(t++, 0, 4));       // valid, nothing due
    s.ops.push_back(op_adv(t++, 3, 4));       // load complete tile1
    s.ops.push_back(op_comp(t++, 0, 1));
    s.ops.push_back(op_adv(t++, 3, 4));
    s.ops.push_back(op_store(t++, 0, 1));
    s.ops.push_back(op_adv(t++, 3, 4));       // tile1 done; buffer0 empty
    s.ops.push_back(op_cancel(t++, 1));       // id1 now DONE -> terminal -> invalid
    s.ops.push_back(op_reset(t++, 0));        // buffer0 empty again -> valid
    return s;
}

// S7: large mixed random stress with clock wrap and adversarial interleaving.
static Scenario sc_mixed_random() {
    Scenario s; s.name = "mixed_random_stress";
    s.spec = make_spec(6, 2, 2, 2, 256, 1024, 256, 4096);
    SplitMix64 rng(g_state ^ 0xc0ffee1234ULL);
    int t = 0;
    uint64_t next_id = 1;
    for (int i = 0; i < 20; ++i) {
        s.ops.push_back(op_enq(t++, next_id++, rng.uniform_int(1, 8),
                               rng.uniform_int(1, 8), rng.uniform_int(1, 8),
                               rng.next_u64()));
    }
    for (int i = 0; i < 700; ++i) {
        int pick = rng.uniform_int(0, 99);
        if (pick < 14) {
            s.ops.push_back(op_enq(t++, next_id++, rng.uniform_int(1, 8),
                                   rng.uniform_int(1, 8), rng.uniform_int(1, 8),
                                   rng.next_u64()));
        } else if (pick < 30) {
            s.ops.push_back(op_load(t++, rng.uniform_int(0, 1), rng.uniform_int(1, 4)));
        } else if (pick < 46) {
            s.ops.push_back(op_comp(t++, rng.uniform_int(0, 1), rng.uniform_int(1, 4)));
        } else if (pick < 62) {
            s.ops.push_back(op_store(t++, rng.uniform_int(0, 1), rng.uniform_int(1, 4)));
        } else if (pick < 86) {
            s.ops.push_back(op_adv(t++, rng.uniform_int(0, 12), rng.uniform_int(1, 6)));
        } else if (pick < 94) {
            s.ops.push_back(op_cancel(t++, (uint64_t)rng.uniform_int(1, (int)next_id)));
        } else {
            s.ops.push_back(op_reset(t++, rng.uniform_int(0, s.spec.barrier_count - 1)));
        }
    }
    // clock wrap then big advances to flush.
    s.ops.push_back(op_adv(t++, 0xfffffffffffffff0ULL, 0));   // wrap, max_async 0 -> no process
    s.ops.push_back(op_adv(t++, 0x100ULL, 64));               // wraps; flush due
    for (int r = 0; r < 60; ++r) {
        s.ops.push_back(op_load(t++, r % 2, 4));
        s.ops.push_back(op_adv(t++, 8, 16));
        s.ops.push_back(op_comp(t++, r % 2, 4));
        s.ops.push_back(op_adv(t++, 8, 16));
        s.ops.push_back(op_store(t++, r % 2, 4));
        s.ops.push_back(op_adv(t++, 8, 16));
    }
    return s;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(sc_basic_pipeline());
    v.push_back(sc_multibuffer());
    v.push_back(sc_mbarrier_waits());
    v.push_back(sc_stale_cancel());
    v.push_back(sc_role_order());
    v.push_back(sc_reset_invalid());
    v.push_back(sc_mixed_random());
    return v;
}

// ------------------------------------------------------------- runner
static bool run_one_op(
    const MkwpProblemSpec& spec, const MkwpRunSpec& op, MkwpOracle* oracle,
    void* state, void* workspace, size_t workspace_bytes, cudaStream_t stream,
    StepResult* result, std::string* error) {
    (void)spec;

    GuardedDeviceBuffer<int64_t> d_counts;
    GuardedDeviceBuffer<int32_t> d_opidx;
    GuardedDeviceBuffer<uint64_t> d_clock, d_eseq, d_pipe, d_buf, d_bar, d_tile, d_async, d_state;
    d_counts.allocate(MKWP_COUNT_N);
    d_opidx.allocate(1);
    d_clock.allocate(1); d_eseq.allocate(1); d_pipe.allocate(1); d_buf.allocate(1);
    d_bar.allocate(1); d_tile.allocate(1); d_async.allocate(1); d_state.allocate(1);

    MkwpOutputs outputs = {};
    outputs.counts = d_counts.ptr;
    outputs.op_index_out = d_opidx.ptr;
    outputs.clock_out = d_clock.ptr;
    outputs.event_seq_out = d_eseq.ptr;
    outputs.pipe_event_hash = d_pipe.ptr;
    outputs.buffer_hash = d_buf.ptr;
    outputs.barrier_hash = d_bar.ptr;
    outputs.tile_hash = d_tile.ptr;
    outputs.async_hash = d_async.ptr;
    outputs.state_checksum = d_state.ptr;

    MkwpRunSpec op_copy = op;  // verify immutability
    MkwpInputs inputs = {};
    inputs.reserved = nullptr;

    CUDA_CHECK(solution_run(state, &op_copy, &inputs, &outputs, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (std::memcmp(&op_copy, &op, sizeof(MkwpRunSpec)) != 0) {
        if (error) *error = "run spec mutated by solution_run";
        return false;
    }

    if (!d_counts.check_guards("counts", error)) return false;
    if (!d_opidx.check_guards("op_index", error)) return false;
    if (!d_clock.check_guards("clock", error)) return false;
    if (!d_eseq.check_guards("event_seq", error)) return false;
    if (!d_pipe.check_guards("pipe_event_hash", error)) return false;
    if (!d_buf.check_guards("buffer_hash", error)) return false;
    if (!d_bar.check_guards("barrier_hash", error)) return false;
    if (!d_tile.check_guards("tile_hash", error)) return false;
    if (!d_async.check_guards("async_hash", error)) return false;
    if (!d_state.check_guards("state_checksum", error)) return false;

    const std::vector<int64_t> h_counts = d_counts.download_data();
    const std::vector<int32_t> h_opidx = d_opidx.download_data();
    const std::vector<uint64_t> h_clock = d_clock.download_data();
    const std::vector<uint64_t> h_eseq = d_eseq.download_data();
    const std::vector<uint64_t> h_pipe = d_pipe.download_data();
    const std::vector<uint64_t> h_buf = d_buf.download_data();
    const std::vector<uint64_t> h_bar = d_bar.download_data();
    const std::vector<uint64_t> h_tile = d_tile.download_data();
    const std::vector<uint64_t> h_async = d_async.download_data();
    const std::vector<uint64_t> h_state = d_state.download_data();

    MkwpExpected expected;
    oracle->step_once(op, &expected);

    MkwpHostOutputsView got = {};
    got.counts = h_counts.data();
    got.op_index_out = h_opidx.data();
    got.clock_out = h_clock.data();
    got.event_seq_out = h_eseq.data();
    got.pipe_event_hash = h_pipe.data();
    got.buffer_hash = h_buf.data();
    got.barrier_hash = h_bar.data();
    got.tile_hash = h_tile.data();
    got.async_hash = h_async.data();
    got.state_checksum = h_state.data();

    if (!mkwp_check_outputs(expected, got, error)) return false;

    if (result) {
        result->counts = h_counts;
        result->op_index = h_opidx[0];
        result->clock = h_clock[0];
        result->event_seq = h_eseq[0];
        result->pipe = h_pipe[0];
        result->buf = h_buf[0];
        result->bar = h_bar[0];
        result->tile = h_tile[0];
        result->async = h_async[0];
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

    MkwpOracle oracle;
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
        if (verbose && (!ok || (i % 128 == 0))) {
            std::printf("scenario %-26s op %04zu/%04zu kind=%d %s%s%s\n",
                        sc.name.c_str(), i, sc.ops.size(), sc.ops[i].op_kind,
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
            a[i].clock != b[i].clock || a[i].event_seq != b[i].event_seq ||
            a[i].pipe != b[i].pipe || a[i].buf != b[i].buf ||
            a[i].bar != b[i].bar || a[i].tile != b[i].tile ||
            a[i].async != b[i].async || a[i].state != b[i].state) {
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
