// file: test_mk_paged_interpreter.cu
//
// 3-way validation harness for MK1 (paged SM interpreter). Drives the device
// solution one op at a time against the CPU oracle, checking exact-integer
// counts and FNV checksums after every op. Includes guard buffers,
// op-spec-immutability checks, and reset + exact-replay checks.

#include "mk_paged_interpreter_common.h"
#include "mk_paged_interpreter_oracle.hpp"

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
    int uniform_int(int lo, int hi) { return lo + (int)(next_u64() % (uint64_t)(hi - lo + 1)); }
    uint64_t uniform_u64(uint64_t lo, uint64_t hi) { return lo + (next_u64() % (hi - lo + 1)); }
};

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr; size_t count = 0;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) { count = n; if (n > 0) CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * n)); }
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
        count = n; data_bytes = sizeof(T) * count;
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
            if (left[i] != kGuardByte) { if (error) { std::ostringstream o; o << "left guard corrupt for " << name << " byte " << i; *error = o.str(); } return false; }
            if (right[i] != kGuardByte) { if (error) { std::ostringstream o; o << "right guard corrupt for " << name << " byte " << i; *error = o.str(); } return false; }
        }
        return true;
    }
};

struct StepResult {
    std::vector<int64_t> counts;
    int32_t op_index = 0;
    uint64_t clock = 0, event_seq = 0, ehash = 0, phash = 0, smhash = 0, chash = 0, pendhash = 0, state = 0;
};

// --------------------------------------------------- program builder
struct ProgramBuilder {
    std::vector<int32_t> program_len;     // [SM]
    std::vector<int32_t> instr_offset;    // [SM]
    std::vector<MkInstr> instrs;
    std::vector<MkPageReq> reqs;
    std::vector<MkWaitRec> waits;
    int SM;
    std::vector<std::vector<MkInstr>> per_sm;            // staging
    std::vector<std::vector<std::vector<MkPageReq>>> per_sm_reqs;
    std::vector<std::vector<std::vector<MkWaitRec>>> per_sm_waits;

    explicit ProgramBuilder(int sm) : SM(sm) {
        per_sm.assign(sm, {});
        per_sm_reqs.assign(sm, {});
        per_sm_waits.assign(sm, {});
    }

    // Add an instruction to SM s. reqs/waits are vectors of (tile,mode,rel) and (counter,target).
    void add(int s, uint64_t instr_id,
             std::vector<MkPageReq> r, std::vector<MkWaitRec> w,
             uint64_t load_lat, uint64_t comp_lat, uint64_t store_lat,
             uint64_t result_seed, uint32_t out_counter, uint64_t out_incr) {
        MkInstr in = {};
        in.instr_id = instr_id;
        in.load_latency = load_lat; in.compute_latency = comp_lat; in.store_latency = store_lat;
        in.result_seed = result_seed; in.out_increment = out_incr;
        in.out_counter = out_counter;
        in.page_req_count = (uint32_t)r.size();
        in.wait_count = (uint32_t)w.size();
        per_sm[s].push_back(in);
        per_sm_reqs[s].push_back(r);
        per_sm_waits[s].push_back(w);
    }

    void finalize() {
        program_len.assign(SM, 0);
        instr_offset.assign(SM, 0);
        instrs.clear(); reqs.clear(); waits.clear();
        for (int s = 0; s < SM; ++s) {
            instr_offset[s] = (int32_t)instrs.size();
            program_len[s] = (int32_t)per_sm[s].size();
            for (size_t k = 0; k < per_sm[s].size(); ++k) {
                MkInstr in = per_sm[s][k];
                in.req_offset = (uint32_t)reqs.size();
                in.wait_offset = (uint32_t)waits.size();
                for (auto& rq : per_sm_reqs[s][k]) reqs.push_back(rq);
                for (auto& wr : per_sm_waits[s][k]) waits.push_back(wr);
                instrs.push_back(in);
            }
        }
    }
};

static MkPageReq mkreq(uint64_t tile, uint8_t mode, uint8_t rel) {
    MkPageReq r = {}; r.tile_id = tile; r.mode = mode; r.release_after_store = rel; return r;
}
static MkWaitRec mkwait(uint32_t ctr, uint64_t target) {
    MkWaitRec w = {}; w.counter_id = ctr; w.target = target; return w;
}

struct Scenario {
    std::string name;
    int SM, PAGES, NCTR, MAXP, MAXPROG;
    ProgramBuilder pb;
    std::vector<MkRunSpec> ops;
    Scenario(const std::string& n, int sm, int pages, int nctr, int maxp, int maxprog)
        : name(n), SM(sm), PAGES(pages), NCTR(nctr), MAXP(maxp), MAXPROG(maxprog), pb(sm) {}
};

static MkProblemSpec make_spec(const Scenario& sc) {
    MkProblemSpec s = {};
    s.abi_version = MK_ABI_VERSION;
    s.sm_count = sc.SM; s.pages_per_sm = sc.PAGES; s.counter_count = sc.NCTR;
    s.max_pending_events = sc.MAXP; s.max_program_len_per_sm = sc.MAXPROG;
    s.total_instr = (int32_t)sc.pb.instrs.size();
    s.total_reqs = (int32_t)sc.pb.reqs.size();
    s.total_waits = (int32_t)sc.pb.waits.size();
    s.program_len = sc.pb.program_len.data();
    s.instr_offset = sc.pb.instr_offset.data();
    s.instrs = sc.pb.instrs.empty() ? nullptr : sc.pb.instrs.data();
    s.reqs = sc.pb.reqs.empty() ? nullptr : sc.pb.reqs.data();
    s.waits = sc.pb.waits.empty() ? nullptr : sc.pb.waits.data();
    return s;
}

// --------------------------------------------------- op constructors
static MkRunSpec mk(int op_kind, int step) {
    MkRunSpec r = {}; r.abi_version = MK_ABI_VERSION; r.op_kind = op_kind; r.step_id = step; return r;
}
static MkRunSpec op_begin(int step, uint64_t pass_id) { MkRunSpec r = mk(MK_OP_BEGIN_PASS, step); r.a_pass_id = pass_id; return r; }
static MkRunSpec op_step(int step, int sm, uint32_t lim) { MkRunSpec r = mk(MK_OP_STEP_SM, step); r.a_sm = sm; r.a_transition_limit = lim; return r; }
static MkRunSpec op_adv(int step, uint64_t delta, uint32_t maxev) { MkRunSpec r = mk(MK_OP_ADVANCE, step); r.a_delta = delta; r.a_max_events = maxev; return r; }
static MkRunSpec op_host(int step, uint32_t ctr, uint64_t amt) { MkRunSpec r = mk(MK_OP_HOST_INC_COUNTER, step); r.a_counter = ctr; r.a_amount = amt; return r; }
static MkRunSpec op_abort(int step) { return mk(MK_OP_ABORT_PASS, step); }

// =================================================================== scenarios

// S1: single SM, single instruction, full happy-path lifecycle with one READ
// page + non-zero latencies. Drives every stage transition once.
static Scenario sc_basic_lifecycle() {
    Scenario s("basic_lifecycle", 1, 4, 2, 16, 8);
    s.pb.add(0, 1001, { mkreq(7, MK_MODE_READ, 1) }, {}, /*load*/5, /*comp*/3, /*store*/2, /*seed*/0xABCD, /*out*/0, /*inc*/1);
    s.pb.finalize();
    int t = 0;
    s.ops.push_back(op_begin(t++, 42));
    s.ops.push_back(op_step(t++, 0, 8));    // FETCH->ACQUIRE(load issue)->LOADING(load_wait)
    s.ops.push_back(op_adv(t++, 5, 4));     // load done at clk 5
    s.ops.push_back(op_step(t++, 0, 8));    // LOADING->WAIT_DEPS->COMPUTE_ISSUE
    s.ops.push_back(op_step(t++, 0, 1));    // COMPUTE_WAIT
    s.ops.push_back(op_adv(t++, 3, 4));     // compute done at clk 8
    s.ops.push_back(op_step(t++, 0, 8));    // STORE_READY->STORE_ISSUE
    s.ops.push_back(op_step(t++, 0, 1));    // STORE_WAIT
    s.ops.push_back(op_adv(t++, 2, 4));     // store done at clk 10 -> release + counter inc + complete
    s.ops.push_back(op_step(t++, 0, 8));    // IDLE -> program done
    s.ops.push_back(op_step(t++, 0, 4));    // DONE -> SM_DONE_WAIT
    return s;
}

// S2: page reuse vs eviction. Two instructions: first loads tile 7 into a page
// and releases (FREE_RESIDENT); second READ of tile 7 reuses; a WRITE of a new
// tile into a resident page evicts.
static Scenario sc_reuse_evict() {
    Scenario s("reuse_evict", 1, 2, 2, 16, 8);
    // i0: READ tile 7, release_after_store=1, inline compute (comp_lat=0).
    s.pb.add(0, 1, { mkreq(7, MK_MODE_READ, 1) }, {}, 2, 0, 1, 0x11, UINT32_MAX, 0);
    // i1: READ tile 7 (reuse of resident page) + SCRATCH tile 99 (held ready).
    s.pb.add(0, 2, { mkreq(7, MK_MODE_READ, 1), mkreq(99, MK_MODE_SCRATCH, 1) }, {}, 2, 0, 1, 0x22, UINT32_MAX, 0);
    // i2: WRITE tile 5 -> must evict a resident page (tile 7 or 99).
    s.pb.add(0, 3, { mkreq(5, MK_MODE_WRITE, 1) }, {}, 2, 0, 1, 0x33, UINT32_MAX, 0);
    s.pb.finalize();
    int t = 0;
    s.ops.push_back(op_begin(t++, 1));
    // drive i0 to completion.
    s.ops.push_back(op_step(t++, 0, 8));   // fetch, acquire(load issue), loading->load_wait
    s.ops.push_back(op_adv(t++, 2, 4));    // load done
    s.ops.push_back(op_step(t++, 0, 8));   // wait_deps -> inline compute -> store_ready -> store_issue
    s.ops.push_back(op_adv(t++, 1, 4));    // store done -> release tile7 page (FREE_RESIDENT), complete
    s.ops.push_back(op_step(t++, 0, 8));   // idle -> fetch i1, acquire: reuse tile7 + scratch tile99
    s.ops.push_back(op_step(t++, 0, 8));   // wait_deps inline -> store ready -> store issue
    s.ops.push_back(op_adv(t++, 1, 8));    // store done -> release both, complete
    s.ops.push_back(op_step(t++, 0, 8));   // fetch i2, acquire WRITE tile5 -> evict resident, load issue
    s.ops.push_back(op_adv(t++, 2, 4));    // load done
    s.ops.push_back(op_step(t++, 0, 8));   // wait_deps inline -> store issue
    s.ops.push_back(op_adv(t++, 1, 8));    // store done -> release, complete
    s.ops.push_back(op_step(t++, 0, 8));   // program done
    return s;
}

// S3: dependency counters — instruction blocks on counter until HOST_INC and a
// producer COUNTER_INC raise it to target. Crux: page is held while blocked.
static Scenario sc_counter_deps() {
    Scenario s("counter_deps", 2, 4, 3, 16, 8);
    // SM0 producer: instr increments counter 1 by 5 on store.
    s.pb.add(0, 100, { mkreq(10, MK_MODE_WRITE, 1) }, {}, 1, 0, 1, 0x1, /*out*/1, /*inc*/5);
    // SM1 consumer: waits counter1>=5 AND counter2>=3 before compute. Holds a SCRATCH page while blocked.
    s.pb.add(1, 200, { mkreq(20, MK_MODE_SCRATCH, 1) }, { mkwait(1, 5), mkwait(2, 3) }, 0, 2, 1, 0x2, UINT32_MAX, 0);
    s.pb.finalize();
    int t = 0;
    s.ops.push_back(op_begin(t++, 7));
    // SM1 first: fetch, acquire SCRATCH (held ready, no load), wait_deps -> counter_wait (counter1=0<5).
    s.ops.push_back(op_step(t++, 1, 8));   // fetch->acquire(scratch)->wait_deps counter_wait
    s.ops.push_back(op_step(t++, 1, 2));   // still counter_wait (page held while blocked)
    // SM0 producer to completion -> counter1 becomes 5.
    s.ops.push_back(op_step(t++, 0, 8));   // fetch, acquire write -> load issue, loading wait
    s.ops.push_back(op_adv(t++, 1, 4));    // load done
    s.ops.push_back(op_step(t++, 0, 8));   // wait_deps inline -> store issue
    s.ops.push_back(op_adv(t++, 1, 4));    // store done -> counter1 += 5 -> 5, complete
    // SM1 now: counter1>=5 but counter2=0<3 -> still counter_wait on second wait.
    s.ops.push_back(op_step(t++, 1, 4));   // counter_wait (counter2)
    s.ops.push_back(op_host(t++, 2, 3));   // host raises counter2 to 3
    s.ops.push_back(op_step(t++, 1, 8));   // all waits satisfied -> compute issue
    s.ops.push_back(op_step(t++, 1, 1));   // compute wait
    s.ops.push_back(op_adv(t++, 2, 4));    // compute done
    s.ops.push_back(op_step(t++, 1, 8));   // store issue
    s.ops.push_back(op_adv(t++, 1, 8));    // store done -> release scratch page, complete
    s.ops.push_back(op_step(t++, 1, 4));   // program done
    s.ops.push_back(op_step(t++, 0, 4));   // SM0 program done
    return s;
}

// S4: page starvation — an instruction reserves all pages then blocks on a
// counter; a later instruction on the SAME SM cannot proceed (it can't, since
// one SM runs one instruction at a time) — instead use page exhaustion ACROSS
// reuse: an instr requesting MORE pages than free -> PAGE_STALL.
static Scenario sc_page_stall() {
    Scenario s("page_stall", 1, 2, 2, 16, 8);
    // i0 requests 3 SCRATCH pages but only 2 exist -> all-or-nothing -> PAGE_STALL forever.
    s.pb.add(0, 1, { mkreq(1, MK_MODE_SCRATCH, 1), mkreq(2, MK_MODE_SCRATCH, 1), mkreq(3, MK_MODE_SCRATCH, 1) }, {}, 1, 1, 1, 0x1, UINT32_MAX, 0);
    s.pb.finalize();
    int t = 0;
    s.ops.push_back(op_begin(t++, 1));
    s.ops.push_back(op_step(t++, 0, 8));   // fetch -> acquire -> PAGE_STALL
    s.ops.push_back(op_step(t++, 0, 4));   // acquire again -> PAGE_STALL (deterministic)
    s.ops.push_back(op_step(t++, 0, 1));   // PAGE_STALL again
    s.ops.push_back(op_abort(t++));        // abort the stuck pass
    s.ops.push_back(op_begin(t++, 2));     // new pass valid after abort
    s.ops.push_back(op_step(t++, 0, 4));   // PAGE_STALL again in pass 2
    s.ops.push_back(op_abort(t++));
    return s;
}

// S5: ABORT + STALE events. Issue loads/computes/stores in flight, then ABORT;
// the in-flight pending events must be dropped in canonical order and pages
// abort-released in (sm,page) order. Then ADVANCE finds stale (already gone).
static Scenario sc_abort_stale() {
    Scenario s("abort_stale", 2, 3, 2, 32, 8);
    s.pb.add(0, 10, { mkreq(1, MK_MODE_READ, 1), mkreq(2, MK_MODE_WRITE, 0) }, {}, 10, 5, 5, 0xA, 0, 1);
    s.pb.add(1, 20, { mkreq(3, MK_MODE_READ, 1) }, {}, 7, 4, 4, 0xB, 1, 2);
    s.pb.finalize();
    int t = 0;
    s.ops.push_back(op_begin(t++, 9));
    s.ops.push_back(op_step(t++, 0, 4));   // SM0 issue two loads (pending)
    s.ops.push_back(op_step(t++, 1, 4));   // SM1 issue one load (pending)
    s.ops.push_back(op_adv(t++, 7, 1));    // process exactly 1 event (lowest due): SM1 load done
    s.ops.push_back(op_step(t++, 1, 8));   // SM1 -> wait_deps inline? comp_lat=4 -> compute issue (pending compute)
    s.ops.push_back(op_abort(t++));        // abort: drop pending (SM0 2 loads + SM1 compute) in canonical order, page abort releases
    s.ops.push_back(op_adv(t++, 100, 16)); // no pending left -> nothing
    s.ops.push_back(op_begin(t++, 10));    // valid restart
    s.ops.push_back(op_step(t++, 0, 4));
    s.ops.push_back(op_step(t++, 1, 4));
    s.ops.push_back(op_abort(t++));
    return s;
}

// S6: multi-SM, multi-instruction interleaved, several counters, mixed
// latencies, ADVANCE with partial max_events (forces canonical tie-break),
// and invalid ops (BEGIN while active+not-done, STEP bad sm, HOST bad counter).
static Scenario sc_interleaved() {
    Scenario s("interleaved_multi_sm", 3, 4, 4, 64, 8);
    for (int sm = 0; sm < 3; ++sm) {
        s.pb.add(sm, 300 + sm * 10 + 0,
                 { mkreq((uint64_t)(sm * 100 + 1), MK_MODE_READ, 1), mkreq((uint64_t)(sm * 100 + 2), MK_MODE_WRITE, 1) },
                 {}, /*load*/ (uint64_t)(3 + sm), /*comp*/ 2, /*store*/ 2, 0x100 + sm, (uint32_t)sm, 1);
        s.pb.add(sm, 300 + sm * 10 + 1,
                 { mkreq((uint64_t)(sm * 100 + 1), MK_MODE_READ, 1) },
                 { mkwait((uint32_t)((sm + 1) % 3), 1) }, /*load*/ 2, /*comp*/ 0, /*store*/ 1, 0x200 + sm, 3, 2);
    }
    s.pb.finalize();
    int t = 0;
    s.ops.push_back(op_begin(t++, 5));
    // fetch + acquire each SM's first instr.
    for (int sm = 0; sm < 3; ++sm) s.ops.push_back(op_step(t++, sm, 4));
    // invalid: BEGIN while active and SMs not DONE.
    s.ops.push_back(op_begin(t++, 99));
    // invalid: STEP bad sm.
    s.ops.push_back(op_step(t++, 9, 2));
    // invalid: HOST bad counter.
    s.ops.push_back(op_host(t++, 50, 1));
    // advance a little to release some loads, partial events.
    s.ops.push_back(op_adv(t++, 3, 1));   // exactly one event (canonical min)
    s.ops.push_back(op_adv(t++, 3, 2));   // two more
    s.ops.push_back(op_adv(t++, 10, 32)); // drain the rest of loads
    // step all SMs forward through compute/store as far as possible.
    for (int round = 0; round < 6; ++round) {
        for (int sm = 0; sm < 3; ++sm) s.ops.push_back(op_step(t++, sm, 4));
        s.ops.push_back(op_adv(t++, 20, 32));
    }
    // host bumps so the dependent second instructions can pass their waits.
    for (uint32_t ci = 0; ci < 4; ++ci) s.ops.push_back(op_host(t++, ci, 10));
    for (int round = 0; round < 8; ++round) {
        for (int sm = 0; sm < 3; ++sm) s.ops.push_back(op_step(t++, sm, 4));
        s.ops.push_back(op_adv(t++, 20, 32));
    }
    // everyone should reach program done.
    for (int sm = 0; sm < 3; ++sm) s.ops.push_back(op_step(t++, sm, 4));
    return s;
}

// S7: big randomized stress over all ops, clock wrap, many SMs/pages/counters.
static Scenario sc_random_stress() {
    Scenario s("random_stress", 4, 6, 6, 256, 8);
    SplitMix64 rng(0xdeadbeefcafe1234ULL);
    uint64_t instr_id = 1;
    for (int sm = 0; sm < 4; ++sm) {
        int n = rng.uniform_int(2, 6);
        for (int k = 0; k < n; ++k) {
            int nreq = rng.uniform_int(0, 3);
            std::vector<MkPageReq> r;
            for (int q = 0; q < nreq; ++q) {
                int mode = rng.uniform_int(0, 2);
                r.push_back(mkreq(rng.uniform_u64(1, 8), (uint8_t)mode, (uint8_t)rng.uniform_int(0, 1)));
            }
            int nwait = rng.uniform_int(0, 2);
            std::vector<MkWaitRec> w;
            for (int q = 0; q < nwait; ++q) w.push_back(mkwait((uint32_t)rng.uniform_int(0, 5), rng.uniform_u64(1, 6)));
            s.pb.add(sm, instr_id++, r, w,
                     rng.uniform_u64(0, 4), rng.uniform_u64(0, 4), rng.uniform_u64(0, 4),
                     rng.next_u64(), rng.uniform_int(0, 1) ? (uint32_t)rng.uniform_int(0, 5) : UINT32_MAX,
                     rng.uniform_u64(1, 3));
        }
    }
    s.pb.finalize();
    int t = 0;
    s.ops.push_back(op_begin(t++, rng.next_u64()));
    for (int i = 0; i < 700; ++i) {
        int pick = rng.uniform_int(0, 99);
        if (pick < 50) {
            s.ops.push_back(op_step(t++, rng.uniform_int(0, 3), (uint32_t)rng.uniform_int(1, 4)));
        } else if (pick < 78) {
            s.ops.push_back(op_adv(t++, rng.uniform_u64(0, 6), (uint32_t)rng.uniform_int(0, 4)));
        } else if (pick < 92) {
            s.ops.push_back(op_host(t++, (uint32_t)rng.uniform_int(0, 6), rng.uniform_u64(1, 4)));
        } else {
            // occasionally abort + restart to exercise stale/abort paths.
            s.ops.push_back(op_abort(t++));
            s.ops.push_back(op_begin(t++, rng.next_u64()));
        }
    }
    // clock wrap near 2^64.
    s.ops.push_back(op_adv(t++, 0xfffffffffffffff0ULL, 8));
    s.ops.push_back(op_adv(t++, 0x100ULL, 32));   // wraps
    // drain.
    for (int round = 0; round < 30; ++round) {
        for (int sm = 0; sm < 4; ++sm) s.ops.push_back(op_step(t++, sm, 4));
        s.ops.push_back(op_adv(t++, 50, 64));
    }
    s.ops.push_back(op_abort(t++));
    return s;
}

static std::vector<Scenario*> build_scenarios() {
    std::vector<Scenario*> v;
    v.push_back(new Scenario(sc_basic_lifecycle()));
    v.push_back(new Scenario(sc_reuse_evict()));
    v.push_back(new Scenario(sc_counter_deps()));
    v.push_back(new Scenario(sc_page_stall()));
    v.push_back(new Scenario(sc_abort_stale()));
    v.push_back(new Scenario(sc_interleaved()));
    v.push_back(new Scenario(sc_random_stress()));
    return v;
}

// --------------------------------------------------- single-op runner
static bool run_one_op(
    const MkProblemSpec& spec, const MkRunSpec& op, MkOracle* oracle,
    void* state, void* workspace, size_t workspace_bytes, cudaStream_t stream,
    StepResult* result, std::string* error) {

    GuardedDeviceBuffer<int64_t> d_counts;
    GuardedDeviceBuffer<int32_t> d_opidx;
    GuardedDeviceBuffer<uint64_t> d_clock, d_eseq, d_eh, d_ph, d_sh, d_ch, d_pend, d_state;
    d_counts.allocate(MK_COUNT_N);
    d_opidx.allocate(1);
    d_clock.allocate(1); d_eseq.allocate(1); d_eh.allocate(1); d_ph.allocate(1);
    d_sh.allocate(1); d_ch.allocate(1); d_pend.allocate(1); d_state.allocate(1);

    MkOutputs outputs = {};
    outputs.counts = d_counts.ptr;
    outputs.op_index_out = d_opidx.ptr;
    outputs.clock_out = d_clock.ptr;
    outputs.event_seq_out = d_eseq.ptr;
    outputs.event_hash = d_eh.ptr;
    outputs.page_hash = d_ph.ptr;
    outputs.sm_hash = d_sh.ptr;
    outputs.counter_hash = d_ch.ptr;
    outputs.pending_hash = d_pend.ptr;
    outputs.state_checksum = d_state.ptr;

    MkRunSpec op_copy = op;   // verify immutability
    MkInputs inputs = {}; inputs.reserved = nullptr;

    CUDA_CHECK(solution_run(state, &op_copy, &inputs, &outputs, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (std::memcmp(&op_copy, &op, sizeof(MkRunSpec)) != 0) {
        if (error) *error = "run spec mutated by solution_run";
        return false;
    }

    if (!d_counts.check_guards("counts", error)) return false;
    if (!d_opidx.check_guards("op_index", error)) return false;
    if (!d_clock.check_guards("clock", error)) return false;
    if (!d_eseq.check_guards("event_seq", error)) return false;
    if (!d_eh.check_guards("event_hash", error)) return false;
    if (!d_ph.check_guards("page_hash", error)) return false;
    if (!d_sh.check_guards("sm_hash", error)) return false;
    if (!d_ch.check_guards("counter_hash", error)) return false;
    if (!d_pend.check_guards("pending_hash", error)) return false;
    if (!d_state.check_guards("state_checksum", error)) return false;

    const std::vector<int64_t> h_counts = d_counts.download_data();
    const std::vector<int32_t> h_opidx = d_opidx.download_data();
    const std::vector<uint64_t> h_clock = d_clock.download_data();
    const std::vector<uint64_t> h_eseq = d_eseq.download_data();
    const std::vector<uint64_t> h_eh = d_eh.download_data();
    const std::vector<uint64_t> h_ph = d_ph.download_data();
    const std::vector<uint64_t> h_sh = d_sh.download_data();
    const std::vector<uint64_t> h_ch = d_ch.download_data();
    const std::vector<uint64_t> h_pend = d_pend.download_data();
    const std::vector<uint64_t> h_state = d_state.download_data();

    MkExpected expected;
    oracle->step_once(op, &expected);

    MkHostOutputsView got = {};
    got.counts = h_counts.data();
    got.op_index_out = h_opidx.data();
    got.clock_out = h_clock.data();
    got.event_seq_out = h_eseq.data();
    got.event_hash = h_eh.data();
    got.page_hash = h_ph.data();
    got.sm_hash = h_sh.data();
    got.counter_hash = h_ch.data();
    got.pending_hash = h_pend.data();
    got.state_checksum = h_state.data();

    if (!mk_check_outputs(expected, got, error)) return false;

    if (result) {
        result->counts = h_counts;
        result->op_index = h_opidx[0];
        result->clock = h_clock[0];
        result->event_seq = h_eseq[0];
        result->ehash = h_eh[0];
        result->phash = h_ph[0];
        result->smhash = h_sh[0];
        result->chash = h_ch[0];
        result->pendhash = h_pend[0];
        result->state = h_state[0];
    }
    return true;
}

static bool run_scenario_once(
    Scenario& sc, bool verbose, std::vector<StepResult>* results,
    int* passed, int* total, std::string* first_error) {

    MkProblemSpec spec = make_spec(sc);
    size_t workspace_bytes = solution_workspace_bytes(&spec);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    void* state = nullptr;
    CUDA_CHECK(solution_init(&spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    MkOracle oracle;
    oracle.init(spec, sc.pb.program_len, sc.pb.instr_offset, sc.pb.instrs, sc.pb.reqs, sc.pb.waits);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) { results->clear(); results->reserve(sc.ops.size()); }

    bool all_ok = true;
    for (size_t i = 0; i < sc.ops.size(); ++i) {
        StepResult result;
        std::string error;
        const bool ok = run_one_op(spec, sc.ops[i], &oracle, state, workspace.ptr, workspace_bytes, stream,
                                   results ? &result : nullptr, &error);
        ++(*total);
        if (ok) ++(*passed);
        else { all_ok = false; if (first_error && first_error->empty()) { std::ostringstream o; o << sc.name << " op " << i << " (kind=" << sc.ops[i].op_kind << "): " << error; *first_error = o.str(); } }
        if (results) results->push_back(result);
        if (verbose && (!ok || (i % 128 == 0))) {
            std::printf("scenario %-22s op %04zu/%04zu kind=%d %s%s%s\n",
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
            a[i].ehash != b[i].ehash || a[i].phash != b[i].phash ||
            a[i].smhash != b[i].smhash || a[i].chash != b[i].chash ||
            a[i].pendhash != b[i].pendhash || a[i].state != b[i].state) {
            if (error) { std::ostringstream o; o << "replay mismatch at op " << i << " state a=0x" << std::hex << a[i].state << " b=0x" << b[i].state; *error = o.str(); }
            return false;
        }
    }
    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        std::vector<Scenario*> scenarios = build_scenarios();

        int passed = 0, total = 0;
        bool all_ok = true;

        for (Scenario* sc : scenarios) {
            std::vector<StepResult> base_results, replay_results;
            std::string error;

            const bool ok_base = run_scenario_once(*sc, true, &base_results, &passed, &total, &error);
            const bool ok_replay = run_scenario_once(*sc, false, &replay_results, &passed, &total, &error);

            if (ok_base && ok_replay) {
                std::string ce;
                if (compare_results(base_results, replay_results, &ce)) {
                    std::printf("scenario %-22s exact replay PASS (%zu ops)\n", sc->name.c_str(), sc->ops.size());
                } else { all_ok = false; std::printf("scenario %-22s exact replay FAIL  %s\n", sc->name.c_str(), ce.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-22s FAIL  %s\n", sc->name.c_str(), error.c_str());
            }
        }

        for (Scenario* sc : scenarios) delete sc;

        std::printf("passed %d / %d\n", passed, total);
        return (all_ok && passed == total) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
