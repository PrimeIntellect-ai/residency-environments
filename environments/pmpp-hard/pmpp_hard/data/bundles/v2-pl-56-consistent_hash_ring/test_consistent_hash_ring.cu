// file: test_consistent_hash_ring.cu

#include "consistent_hash_ring_common.h"
#include "consistent_hash_ring_oracle.hpp"

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
        if (_err != cudaSuccess) {                                             \
            std::ostringstream _oss;                                            \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                 << cudaGetErrorString(_err);                                  \
            throw std::runtime_error(_oss.str());                              \
        }                                                                       \
    } while (0)

template <typename T>
struct GuardedBuffer {
    uint8_t* raw = nullptr;
    T* ptr = nullptr;
    size_t count = 0;
    size_t data_bytes = 0;
    GuardedBuffer() = default;
    GuardedBuffer(const GuardedBuffer&) = delete;
    GuardedBuffer& operator=(const GuardedBuffer&) = delete;
    ~GuardedBuffer() { if (raw) cudaFree(raw); }
    void allocate(size_t n) {
        count = n; data_bytes = sizeof(T) * n;
        CUDA_CHECK(cudaMalloc((void**)&raw, kGuardBytes + data_bytes + kGuardBytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, kGuardBytes + data_bytes + kGuardBytes));
        ptr = reinterpret_cast<T*>(raw + kGuardBytes);
    }
    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count > 0) CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        return host;
    }
    bool check_guards(const char* name, std::string* err) const {
        std::vector<uint8_t> l(kGuardBytes), r(kGuardBytes);
        CUDA_CHECK(cudaMemcpy(l.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (l[i] != kGuardByte) { if (err) { std::ostringstream o; o << "left guard corrupt for " << name << " @" << i; *err = o.str(); } return false; }
            if (r[i] != kGuardByte) { if (err) { std::ostringstream o; o << "right guard corrupt for " << name << " @" << i; *err = o.str(); } return false; }
        }
        return true;
    }
};

struct Op {
    int32_t op_type;
    int64_t a, b, c;
};

struct Scenario {
    std::string name;
    ChrProblemSpec spec;
    std::vector<Op> ops;
};

static ChrProblemSpec make_spec(int rf, int extra, uint64_t ring_seed, uint64_t key_seed, uint64_t defcap) {
    ChrProblemSpec s = {};
    s.abi_version = CHR_ABI_VERSION;
    s.replication_factor = rf;
    s.preference_list_extra = extra;
    s.max_nodes = 64;
    s.max_vnodes = 1024;
    s.max_keys = 256;
    s.max_replicas_per_key = 8;
    s.max_move_tasks = 4096;
    s.default_node_capacity = defcap;
    s.ring_seed = ring_seed;
    s.key_seed = key_seed;
    return s;
}

static void op(Scenario& sc, int32_t t, int64_t a = 0, int64_t b = 0, int64_t c = 0) {
    Op o; o.op_type = t; o.a = a; o.b = b; o.c = c; sc.ops.push_back(o);
}

// ---- Scenario 1: basic join/activate/put/lookup ----
static Scenario sc_basic() {
    Scenario sc; sc.name = "basic_join_put_lookup";
    sc.spec = make_spec(3, 1, 0x9e3779b97f4a7c15ULL, 0xc2b2ae3d27d4eb4fULL, 100);
    for (int n = 1; n <= 5; ++n) { op(sc, CHR_OP_ADD_NODE, n, 8, 0); op(sc, CHR_OP_ACTIVATE_NODE, n); }
    for (int k = 100; k < 120; ++k) op(sc, CHR_OP_PUT_KEY, k, k * 7 + 1);
    for (int k = 100; k < 120; ++k) op(sc, CHR_OP_LOOKUP, 9000 + k, k);
    op(sc, CHR_OP_REBALANCE, 1000);
    for (int k = 100; k < 110; ++k) op(sc, CHR_OP_LOOKUP, 8000 + k, k);
    return sc;
}

// ---- Scenario 2: leave with rebalance ----
static Scenario sc_leave() {
    Scenario sc; sc.name = "leave_rebalance";
    sc.spec = make_spec(3, 2, 0x1234567890abcdefULL, 0xfedcba0987654321ULL, 50);
    for (int n = 1; n <= 6; ++n) { op(sc, CHR_OP_ADD_NODE, n, 6, 0); op(sc, CHR_OP_ACTIVATE_NODE, n); }
    for (int k = 1; k <= 30; ++k) op(sc, CHR_OP_PUT_KEY, k, k);
    op(sc, CHR_OP_REBALANCE, 100);
    op(sc, CHR_OP_START_LEAVE, 2);
    op(sc, CHR_OP_REBALANCE, 100);
    op(sc, CHR_OP_REMOVE_NODE, 2);   // may stall (serving replicas remain) -> later
    op(sc, CHR_OP_REBALANCE, 100);
    op(sc, CHR_OP_REMOVE_NODE, 2);
    for (int k = 1; k <= 10; ++k) op(sc, CHR_OP_LOOKUP, 5000 + k, k);
    return sc;
}

// ---- Scenario 3: fail / hinted handoff / recover ----
static Scenario sc_fail_recover() {
    Scenario sc; sc.name = "fail_hint_recover";
    sc.spec = make_spec(3, 2, 0xa5a5a5a5deadbeefULL, 0x0123456789abcdefULL, 40);
    for (int n = 1; n <= 6; ++n) { op(sc, CHR_OP_ADD_NODE, n, 5, 0); op(sc, CHR_OP_ACTIVATE_NODE, n); }
    for (int k = 1; k <= 24; ++k) op(sc, CHR_OP_PUT_KEY, k, k * 3);
    op(sc, CHR_OP_REBALANCE, 200);
    op(sc, CHR_OP_FAIL_NODE, 3);
    op(sc, CHR_OP_REBALANCE, 200);   // hinted handoff installs
    for (int k = 1; k <= 8; ++k) op(sc, CHR_OP_LOOKUP, 7000 + k, k);
    op(sc, CHR_OP_RECOVER_NODE, 3);
    op(sc, CHR_OP_REBALANCE, 200);   // handoff back to recovered
    for (int k = 1; k <= 8; ++k) op(sc, CHR_OP_LOOKUP, 7700 + k, k);
    return sc;
}

// ---- Scenario 4: capacity pressure (tight capacity forces stalls) ----
static Scenario sc_capacity() {
    Scenario sc; sc.name = "capacity_stalls";
    sc.spec = make_spec(3, 1, 0xbeefcafe12345678ULL, 0x8765432187654321ULL, 4); // tiny cap
    for (int n = 1; n <= 4; ++n) { op(sc, CHR_OP_ADD_NODE, n, 4, 0); op(sc, CHR_OP_ACTIVATE_NODE, n); }
    for (int k = 1; k <= 20; ++k) op(sc, CHR_OP_PUT_KEY, k, k);
    op(sc, CHR_OP_REBALANCE, 5);
    op(sc, CHR_OP_REBALANCE, 5);
    op(sc, CHR_OP_ADD_NODE, 9, 4, 50); op(sc, CHR_OP_ACTIVATE_NODE, 9); // capacious node
    op(sc, CHR_OP_REBALANCE, 100);
    for (int k = 1; k <= 20; ++k) op(sc, CHR_OP_LOOKUP, 6000 + k, k);
    return sc;
}

// ---- Scenario 5: put/delete/resurrect + obsolete moves via version bump ----
static Scenario sc_versioning() {
    Scenario sc; sc.name = "delete_resurrect_obsolete";
    sc.spec = make_spec(2, 2, 0x11aa22bb33cc44ddULL, 0x55ee66ff77001122ULL, 30);
    for (int n = 1; n <= 5; ++n) { op(sc, CHR_OP_ADD_NODE, n, 6, 0); op(sc, CHR_OP_ACTIVATE_NODE, n); }
    for (int k = 1; k <= 15; ++k) op(sc, CHR_OP_PUT_KEY, k, k);
    op(sc, CHR_OP_START_LEAVE, 1);        // enqueue add tasks at v
    op(sc, CHR_OP_PUT_KEY, 1, 999);       // bump version -> obsolete some tasks
    op(sc, CHR_OP_DELETE_KEY, 5);         // delete -> delete replicas + obsolete
    op(sc, CHR_OP_REBALANCE, 50);
    op(sc, CHR_OP_PUT_KEY, 5, 5050);      // resurrect
    op(sc, CHR_OP_REBALANCE, 50);
    op(sc, CHR_OP_LOOKUP, 1, 1);
    op(sc, CHR_OP_LOOKUP, 5, 5);
    op(sc, CHR_OP_DELETE_KEY, 99);        // miss
    op(sc, CHR_OP_LOOKUP, 7, 99);         // missing

    // Guaranteed MOVE_OBSOLETE coverage: FAIL a node (enqueues HINTED add tasks at
    // the keys' current versions), then bump every key's version via PUT before
    // rebalancing, so those pending tasks are invalidated by version mismatch.
    op(sc, CHR_OP_FAIL_NODE, 3);
    for (int k = 1; k <= 15; ++k) op(sc, CHR_OP_PUT_KEY, k, 70000 + k);
    op(sc, CHR_OP_REBALANCE, 200);        // pending FAIL tasks now obsolete
    for (int k = 1; k <= 15; ++k) op(sc, CHR_OP_LOOKUP, 4000 + k, k);
    return sc;
}

// ---- Scenario 6: adversarial invalid ops + token collisions ----
static Scenario sc_adversarial() {
    Scenario sc; sc.name = "adversarial_invalids_collisions";
    // small ring_seed range encourages collisions across ordinals; many vnodes.
    sc.spec = make_spec(3, 3, 7, 13, 20);
    op(sc, CHR_OP_ACTIVATE_NODE, 1);      // invalid (absent)
    op(sc, CHR_OP_FAIL_NODE, 1);          // invalid
    op(sc, CHR_OP_REMOVE_NODE, 1);        // invalid
    op(sc, CHR_OP_ADD_NODE, 1, 0, 0);     // invalid vnode_count 0
    op(sc, CHR_OP_ADD_NODE, 1, 16, 0);    // ok
    op(sc, CHR_OP_ADD_NODE, 1, 8, 0);     // invalid (exists)
    op(sc, CHR_OP_ACTIVATE_NODE, 1);
    op(sc, CHR_OP_ACTIVATE_NODE, 1);      // invalid (already active)
    for (int n = 2; n <= 5; ++n) { op(sc, CHR_OP_ADD_NODE, n, 16, 0); op(sc, CHR_OP_ACTIVATE_NODE, n); }
    for (int k = 1; k <= 40; ++k) op(sc, CHR_OP_PUT_KEY, k, k * 2);
    op(sc, CHR_OP_REBALANCE, 0);          // no-op
    op(sc, CHR_OP_REBALANCE, 1000);
    op(sc, CHR_OP_FAIL_NODE, 2);
    op(sc, CHR_OP_FAIL_NODE, 2);          // invalid (already down)
    op(sc, CHR_OP_RECOVER_NODE, 5);       // invalid (not down)
    op(sc, CHR_OP_RECOVER_NODE, 2);
    op(sc, CHR_OP_REBALANCE, 1000);
    op(sc, CHR_OP_START_LEAVE, 3);
    op(sc, CHR_OP_REBALANCE, 1000);
    op(sc, CHR_OP_REMOVE_NODE, 3);
    for (int k = 1; k <= 40; ++k) op(sc, CHR_OP_LOOKUP, 1000 + k, k);
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(sc_basic());
    v.push_back(sc_leave());
    v.push_back(sc_fail_recover());
    v.push_back(sc_capacity());
    v.push_back(sc_versioning());
    v.push_back(sc_adversarial());
    return v;
}

struct OpSnapshot {
    ChrCounters counters;
    uint64_t ring_event_hash, lookup_hash, ring_hash, node_hash, key_replica_hash, move_hash;
};

static bool run_once(const Scenario& sc, bool verbose, std::vector<OpSnapshot>* snaps,
                     int* passed, int* total, std::string* first_err) {
    ChrProblemSpec spec = sc.spec;

    size_t workspace_bytes = solution_workspace_bytes(&spec);
    // MANDATE: clamp only; never fail-guard on workspace_bytes == 0.

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint8_t* d_workspace = nullptr;
    size_t alloc_ws = std::max<size_t>(workspace_bytes, 1);
    CUDA_CHECK(cudaMalloc((void**)&d_workspace, alloc_ws));

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ChrOracle oracle;
    oracle.init(spec);

    if (snaps) { snaps->clear(); snaps->reserve(sc.ops.size()); }

    bool all_ok = true;

    for (size_t i = 0; i < sc.ops.size(); ++i) {
        const Op& o = sc.ops[i];

        ChrRunSpec run = {};
        run.abi_version = CHR_ABI_VERSION;
        run.op_type = o.op_type;
        run.op_index = (int32_t)i;
        run.arg_a = o.a;
        run.arg_b = o.b;
        run.arg_c = o.c;

        GuardedBuffer<ChrCounters> d_ctr;
        GuardedBuffer<uint64_t> d_evh, d_lkh, d_rh, d_nh, d_kr, d_mh;
        d_ctr.allocate(1); d_evh.allocate(1); d_lkh.allocate(1);
        d_rh.allocate(1); d_nh.allocate(1); d_kr.allocate(1); d_mh.allocate(1);

        ChrOutputs outputs = {};
        outputs.counters = d_ctr.ptr;
        outputs.ring_event_hash = d_evh.ptr;
        outputs.lookup_hash = d_lkh.ptr;
        outputs.ring_hash = d_rh.ptr;
        outputs.node_hash = d_nh.ptr;
        outputs.key_replica_hash = d_kr.ptr;
        outputs.move_hash = d_mh.ptr;

        CUDA_CHECK(solution_run(state, &run, nullptr, &outputs, d_workspace, workspace_bytes, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::string error;
        bool ok = true;
        ok = ok && d_ctr.check_guards("counters", &error);
        ok = ok && d_evh.check_guards("ring_event_hash", &error);
        ok = ok && d_lkh.check_guards("lookup_hash", &error);
        ok = ok && d_rh.check_guards("ring_hash", &error);
        ok = ok && d_nh.check_guards("node_hash", &error);
        ok = ok && d_kr.check_guards("key_replica_hash", &error);
        ok = ok && d_mh.check_guards("move_hash", &error);

        std::vector<ChrCounters> h_ctr = d_ctr.download();
        std::vector<uint64_t> h_evh = d_evh.download(), h_lkh = d_lkh.download();
        std::vector<uint64_t> h_rh = d_rh.download(), h_nh = d_nh.download();
        std::vector<uint64_t> h_kr = d_kr.download(), h_mh = d_mh.download();

        ChrExpected expected;
        chr_oracle_step(oracle, run, &expected);

        ChrGotView got = {};
        got.counters = h_ctr.data();
        got.ring_event_hash = h_evh.data();
        got.lookup_hash = h_lkh.data();
        got.ring_hash = h_rh.data();
        got.node_hash = h_nh.data();
        got.key_replica_hash = h_kr.data();
        got.move_hash = h_mh.data();

        ok = ok && chr_check_outputs(expected, got, &error);

        OpSnapshot snap;
        snap.counters = h_ctr[0];
        snap.ring_event_hash = h_evh[0];
        snap.lookup_hash = h_lkh[0];
        snap.ring_hash = h_rh[0];
        snap.node_hash = h_nh[0];
        snap.key_replica_hash = h_kr[0];
        snap.move_hash = h_mh[0];

        ++(*total);
        if (ok) ++(*passed);
        else {
            all_ok = false;
            if (first_err && first_err->empty()) {
                std::ostringstream oss;
                oss << sc.name << " op " << i << " (type " << o.op_type << "): " << error;
                *first_err = oss.str();
            }
        }

        if (snaps) snaps->push_back(snap);

        if (verbose && (!ok || (i % 25 == 0))) {
            std::printf("  %-30s op %03zu/%03zu type=%d %s%s\n",
                        sc.name.c_str(), i, sc.ops.size(), o.op_type,
                        ok ? "PASS" : "FAIL ", ok ? "" : error.c_str());
        }
    }

    if (verbose && !snaps->empty()) {
        const ChrCounters& c = snaps->back().counters;
        std::printf("    [final] enq=%llu stall=%llu obsolete=%llu radd=%llu rdrop=%llu "
                    "hint_add=%llu hint_drop=%llu rdel=%llu direct=%llu lk_found=%llu "
                    "lk_miss=%llu remove_stall=%llu no_target=%llu invalid=%llu\n",
                    (unsigned long long)c.move_tasks_enqueued, (unsigned long long)c.move_stall,
                    (unsigned long long)c.move_obsolete, (unsigned long long)c.replica_added,
                    (unsigned long long)c.replica_dropped, (unsigned long long)c.hint_handoff_added,
                    (unsigned long long)c.hint_dropped, (unsigned long long)c.replica_deleted,
                    (unsigned long long)c.direct_replica_added, (unsigned long long)c.lookup_found_replicas,
                    (unsigned long long)c.lookup_missing, (unsigned long long)c.remove_stalls,
                    (unsigned long long)c.no_target_count, (unsigned long long)c.invalid_count);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    cudaFree(d_workspace);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_snaps(const std::vector<OpSnapshot>& a, const std::vector<OpSnapshot>& b, std::string* err) {
    if (a.size() != b.size()) { if (err) *err = "op count mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i].counters, &b[i].counters, sizeof(ChrCounters)) != 0 ||
            a[i].ring_event_hash != b[i].ring_event_hash ||
            a[i].lookup_hash != b[i].lookup_hash ||
            a[i].ring_hash != b[i].ring_hash ||
            a[i].node_hash != b[i].node_hash ||
            a[i].key_replica_hash != b[i].key_replica_hash ||
            a[i].move_hash != b[i].move_hash) {
            if (err) { std::ostringstream o; o << "replay mismatch at op " << i; *err = o.str(); }
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
            std::vector<OpSnapshot> base, replay;
            std::string err;
            bool ok_base = run_once(sc, true, &base, &passed, &total, &err);
            bool ok_replay = run_once(sc, false, &replay, &passed, &total, &err);

            if (ok_base && ok_replay) {
                std::string cerr;
                if (compare_snaps(base, replay, &cerr))
                    std::printf("scenario %-30s exact replay PASS (%zu ops)\n", sc.name.c_str(), sc.ops.size());
                else { all_ok = false; std::printf("scenario %-30s replay FAIL  %s\n", sc.name.c_str(), cerr.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-30s FAIL  %s\n", sc.name.c_str(), err.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return (all_ok && passed == total) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
