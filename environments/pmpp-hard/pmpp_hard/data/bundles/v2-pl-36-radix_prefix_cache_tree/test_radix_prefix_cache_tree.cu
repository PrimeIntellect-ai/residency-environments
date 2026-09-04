// file: test_radix_prefix_cache_tree.cu

#include "radix_prefix_cache_tree_common.h"
#include "radix_prefix_cache_tree_oracle.hpp"

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
                 << cudaGetErrorString(_err);                                   \
            throw std::runtime_error(_oss.str());                              \
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

struct OpHost {
    int32_t kind;
    int32_t arg_a;
    int32_t arg_b;
    std::vector<int32_t> tokens;  // only for INSERT
};

struct StepHost {
    std::vector<OpHost> ops;
};

struct Scenario {
    std::string name;
    std::vector<StepHost> steps;
};

static OpHost mk_insert(int32_t req, std::vector<int32_t> toks) {
    OpHost o; o.kind = RPCT_OP_INSERT; o.arg_a = req; o.arg_b = 0; o.tokens = std::move(toks); return o;
}
static OpHost mk_release(int32_t req) {
    OpHost o; o.kind = RPCT_OP_RELEASE; o.arg_a = req; o.arg_b = 0; return o;
}
static OpHost mk_evict(int32_t target) {
    OpHost o; o.kind = RPCT_OP_EVICT; o.arg_a = target; o.arg_b = 0; return o;
}

// ---- scenario builders ----

// 1: basic shared-prefix inserts producing splits.
static Scenario sc_basic_split() {
    Scenario sc; sc.name = "basic_split";
    StepHost s;
    s.ops.push_back(mk_insert(0, {1, 2, 3, 4, 5}));
    s.ops.push_back(mk_insert(1, {1, 2, 3, 9, 9}));  // splits after [1,2,3]
    s.ops.push_back(mk_insert(2, {1, 2, 7}));        // splits [1,2,3]->[1,2]+[3]
    s.ops.push_back(mk_insert(3, {1, 2, 3, 4, 5}));  // full existing prefix
    s.ops.push_back(mk_insert(4, {8, 8, 8}));        // disjoint
    sc.steps.push_back(s);
    return sc;
}

// 2: insert then release then evict (referenced-node protection).
static Scenario sc_release_evict() {
    Scenario sc; sc.name = "release_evict";
    StepHost s;
    s.ops.push_back(mk_insert(10, {1, 2, 3}));
    s.ops.push_back(mk_insert(11, {1, 2, 4}));    // split
    s.ops.push_back(mk_evict(100));               // nothing evictable (all ref'd)
    s.ops.push_back(mk_release(10));              // free path of req10
    s.ops.push_back(mk_evict(1));                 // evict leaf [3] (req10 leaf)
    s.ops.push_back(mk_release(11));
    s.ops.push_back(mk_evict(1000));              // evict remaining unreferenced
    sc.steps.push_back(s);
    return sc;
}

// 3: LRU ordering + tie-break by id.
static Scenario sc_lru_ties() {
    Scenario sc; sc.name = "lru_ties";
    StepHost s;
    // three disjoint single-token leaves inserted in order -> ids 1,2,3, ts 1,2,3
    s.ops.push_back(mk_insert(0, {100}));
    s.ops.push_back(mk_insert(1, {200}));
    s.ops.push_back(mk_insert(2, {300}));
    s.ops.push_back(mk_release(0));
    s.ops.push_back(mk_release(1));
    s.ops.push_back(mk_release(2));
    // evict 1 token: oldest ts is node1 (ts=1). Then evict again -> node2.
    s.ops.push_back(mk_evict(1));
    s.ops.push_back(mk_evict(1));
    s.ops.push_back(mk_evict(1));
    sc.steps.push_back(s);
    return sc;
}

// 4: deep chain split cascade and re-insertion across steps (persistent state).
static Scenario sc_persistent_multistep() {
    Scenario sc; sc.name = "persistent_multistep";
    StepHost s1;
    s1.ops.push_back(mk_insert(0, {5, 6, 7, 8, 9, 10}));
    s1.ops.push_back(mk_insert(1, {5, 6, 7}));        // split mid-edge
    sc.steps.push_back(s1);
    StepHost s2;
    s2.ops.push_back(mk_insert(2, {5, 6, 11, 12}));   // split [5,6,7]->[5,6]+[7]
    s2.ops.push_back(mk_release(0));
    s2.ops.push_back(mk_insert(3, {5, 6, 7, 8, 9, 10}));  // re-grab freed path
    sc.steps.push_back(s2);
    StepHost s3;
    s3.ops.push_back(mk_release(1));
    s3.ops.push_back(mk_release(2));
    s3.ops.push_back(mk_release(3));
    s3.ops.push_back(mk_evict(1000000));   // evict whole tree
    sc.steps.push_back(s3);
    return sc;
}

// 5: cascade eviction where deleting a leaf exposes parent leaf in same op.
static Scenario sc_cascade_evict() {
    Scenario sc; sc.name = "cascade_evict";
    StepHost s;
    s.ops.push_back(mk_insert(0, {1, 2, 3, 4}));      // chain
    s.ops.push_back(mk_insert(1, {1, 2, 3, 4, 5, 6})); // extend -> split makes [..4] internal
    s.ops.push_back(mk_release(0));
    s.ops.push_back(mk_release(1));
    // now: node [1,2,3,4] internal (ref 0 after releases), child [5,6] leaf.
    // evicting a large target should remove leaf then its parent in one op.
    s.ops.push_back(mk_evict(1000));
    sc.steps.push_back(s);
    return sc;
}

// 6: pseudo-random workload mixing inserts/releases/evicts and re-used reqs.
static Scenario sc_random_mix() {
    Scenario sc; sc.name = "random_mix";
    uint64_t rng = 0x12345678ABCDEF01ULL;
    auto next = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };

    const int kVocab = 6;
    for (int step = 0; step < 6; ++step) {
        StepHost s;
        const int ops = 6 + (int)(next() % 6);
        for (int o = 0; o < ops; ++o) {
            const int r = (int)(next() % 100);
            if (r < 55) {
                int req = (int)(next() % 12);
                int len = 1 + (int)(next() % 6);
                std::vector<int32_t> toks;
                for (int t = 0; t < len; ++t) toks.push_back((int32_t)(next() % kVocab));
                s.ops.push_back(mk_insert(req, toks));
            } else if (r < 80) {
                int req = (int)(next() % 12);
                s.ops.push_back(mk_release(req));
            } else {
                int target = (int)(next() % 8);
                s.ops.push_back(mk_evict(target));
            }
        }
        sc.steps.push_back(s);
    }
    return sc;
}

// 7: large fan + repeated identical inserts (idempotent re-grab, ref accumulation).
static Scenario sc_fanout_repeat() {
    Scenario sc; sc.name = "fanout_repeat";
    StepHost s;
    for (int i = 0; i < 24; ++i) {
        // common prefix [7,7] then divergent token i
        s.ops.push_back(mk_insert(i, {7, 7, (int32_t)(1000 + i)}));
    }
    // re-insert same sequences under new request ids -> ref counts climb, no new nodes
    for (int i = 0; i < 24; ++i) {
        s.ops.push_back(mk_insert(100 + i, {7, 7, (int32_t)(1000 + i)}));
    }
    // release half, evict aggressively
    for (int i = 0; i < 24; ++i) s.ops.push_back(mk_release(i));
    s.ops.push_back(mk_evict(5));
    s.ops.push_back(mk_evict(100000));
    sc.steps.push_back(s);
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(sc_basic_split());
    v.push_back(sc_release_evict());
    v.push_back(sc_lru_ties());
    v.push_back(sc_persistent_multistep());
    v.push_back(sc_cascade_evict());
    v.push_back(sc_random_mix());
    v.push_back(sc_fanout_repeat());
    return v;
}

// flatten a step's ops into the device input layout.
struct FlatStep {
    std::vector<int32_t> op_kind;
    std::vector<int32_t> op_arg_a;
    std::vector<int32_t> op_arg_b;
    std::vector<int32_t> op_token_offset;
    std::vector<int32_t> op_token_len;
    std::vector<int32_t> op_tokens;
};

static FlatStep flatten(const StepHost& s) {
    FlatStep f;
    int32_t off = 0;
    for (const OpHost& o : s.ops) {
        f.op_kind.push_back(o.kind);
        f.op_arg_a.push_back(o.arg_a);
        f.op_arg_b.push_back(o.arg_b);
        f.op_token_offset.push_back(off);
        f.op_token_len.push_back((int32_t)o.tokens.size());
        for (int32_t t : o.tokens) f.op_tokens.push_back(t);
        off += (int32_t)o.tokens.size();
    }
    return f;
}

// compute spec bounds for a scenario.
static void scenario_bounds(const Scenario& sc, RpctProblemSpec* spec) {
    int max_ops = 0;
    long long total_insert_tokens = 0;
    int max_op_tokens = 0;
    int max_req = 1;
    for (const StepHost& s : sc.steps) {
        max_ops = std::max(max_ops, (int)s.ops.size());
        int step_tokens = 0;
        for (const OpHost& o : s.ops) {
            if (o.kind == RPCT_OP_INSERT) {
                total_insert_tokens += (long long)o.tokens.size();
                step_tokens += (int)o.tokens.size();
                max_req = std::max(max_req, o.arg_a + 1);
            } else if (o.kind == RPCT_OP_RELEASE) {
                max_req = std::max(max_req, o.arg_a + 1);
            }
        }
        max_op_tokens = std::max(max_op_tokens, step_tokens);
    }
    spec->abi_version = RPCT_ABI_VERSION;
    // node capacity: each insert can add at most 2 nodes (split + leaf).
    long long node_budget = 2;
    for (const StepHost& s : sc.steps)
        for (const OpHost& o : s.ops)
            if (o.kind == RPCT_OP_INSERT) node_budget += 2;
    spec->max_nodes = (int32_t)std::max<long long>(node_budget + 8, RPCT_MIN_NODES);
    // token pool budget for reference: leaf appends across the whole run, which
    // is bounded by total inserted tokens. Add headroom.
    spec->max_tokens = (int32_t)std::max<long long>(total_insert_tokens * 2 + 64, 64);
    spec->max_ops = std::max(max_ops, 1);
    spec->max_requests = std::max(max_req, 1);
    spec->max_op_tokens = std::max(max_op_tokens, 1);
}

struct OpResult {
    int32_t matched_prefix_len;
    int32_t num_nodes;
    int32_t num_tokens_cached;
    int32_t num_evicted_nodes;
    int32_t evicted_tokens;
    uint64_t tree_checksum;
    uint64_t state_checksum;
};

static bool run_scenario_once(
    const Scenario& sc,
    bool verbose,
    std::vector<OpResult>* results,
    int* passed,
    int* total,
    std::string* first_error) {
    RpctProblemSpec spec = {};
    scenario_bounds(sc, &spec);

    if (!rpct_validate_problem_spec(&spec)) {
        if (first_error) *first_error = "invalid problem spec";
        return false;
    }

    size_t workspace_bytes = solution_workspace_bytes(&spec);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    RpctOracleState oracle;
    oracle.init(spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) results->clear();

    bool all_ok = true;

    for (size_t si = 0; si < sc.steps.size(); ++si) {
        const StepHost& step = sc.steps[si];
        FlatStep flat = flatten(step);
        const int op_count = (int)step.ops.size();

        RpctRunSpec run = {};
        run.abi_version = RPCT_ABI_VERSION;
        run.op_count = op_count;
        run.total_op_tokens = (int32_t)flat.op_tokens.size();
        run.step_id = (int32_t)si;
        run.flags = 0;
        if (!rpct_validate_run_spec(&run)) throw std::runtime_error("invalid run spec");

        DeviceBuffer<int32_t> d_kind, d_arg_a, d_arg_b, d_off, d_len, d_tokens;
        d_kind.allocate(std::max<size_t>(flat.op_kind.size(), 1));
        d_arg_a.allocate(std::max<size_t>(flat.op_arg_a.size(), 1));
        d_arg_b.allocate(std::max<size_t>(flat.op_arg_b.size(), 1));
        d_off.allocate(std::max<size_t>(flat.op_token_offset.size(), 1));
        d_len.allocate(std::max<size_t>(flat.op_token_len.size(), 1));
        d_tokens.allocate(std::max<size_t>(flat.op_tokens.size(), 1));

        // pad host vectors to allocation size for upload
        auto pad = [](std::vector<int32_t> v, size_t n) { v.resize(n, 0); return v; };
        d_kind.upload(pad(flat.op_kind, d_kind.count));
        d_arg_a.upload(pad(flat.op_arg_a, d_arg_a.count));
        d_arg_b.upload(pad(flat.op_arg_b, d_arg_b.count));
        d_off.upload(pad(flat.op_token_offset, d_off.count));
        d_len.upload(pad(flat.op_token_len, d_len.count));
        d_tokens.upload(pad(flat.op_tokens, d_tokens.count));

        GuardedDeviceBuffer<int32_t> g_matched, g_nnodes, g_ntok, g_envnodes, g_evtok;
        GuardedDeviceBuffer<uint64_t> g_tree, g_state;
        g_matched.allocate((size_t)op_count);
        g_nnodes.allocate((size_t)op_count);
        g_ntok.allocate((size_t)op_count);
        g_envnodes.allocate((size_t)op_count);
        g_evtok.allocate((size_t)op_count);
        g_tree.allocate((size_t)op_count);
        g_state.allocate((size_t)op_count);

        RpctInputs inputs = {};
        inputs.op_kind = d_kind.ptr;
        inputs.op_arg_a = d_arg_a.ptr;
        inputs.op_arg_b = d_arg_b.ptr;
        inputs.op_token_offset = d_off.ptr;
        inputs.op_token_len = d_len.ptr;
        inputs.op_tokens = d_tokens.ptr;

        RpctOutputs outputs = {};
        outputs.matched_prefix_len = g_matched.ptr;
        outputs.num_nodes = g_nnodes.ptr;
        outputs.num_tokens_cached = g_ntok.ptr;
        outputs.num_evicted_nodes = g_envnodes.ptr;
        outputs.evicted_tokens = g_evtok.ptr;
        outputs.tree_checksum = g_tree.ptr;
        outputs.state_checksum = g_state.ptr;

        std::string error;

        CUDA_CHECK(solution_run(state, &run, &inputs, &outputs,
                                workspace.ptr, std::max<size_t>(workspace_bytes, 1), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        bool ok = true;
        ok = ok && g_matched.check_guards("matched_prefix_len", &error);
        ok = ok && g_nnodes.check_guards("num_nodes", &error);
        ok = ok && g_ntok.check_guards("num_tokens_cached", &error);
        ok = ok && g_envnodes.check_guards("num_evicted_nodes", &error);
        ok = ok && g_evtok.check_guards("evicted_tokens", &error);
        ok = ok && g_tree.check_guards("tree_checksum", &error);
        ok = ok && g_state.check_guards("state_checksum", &error);

        // verify inputs unchanged
        if (ok) {
            if (d_kind.download() != pad(flat.op_kind, d_kind.count)) { ok = false; error = "op_kind modified"; }
            if (ok && d_tokens.download() != pad(flat.op_tokens, d_tokens.count)) { ok = false; error = "op_tokens modified"; }
        }

        std::vector<int32_t> h_matched = g_matched.download_data();
        std::vector<int32_t> h_nnodes = g_nnodes.download_data();
        std::vector<int32_t> h_ntok = g_ntok.download_data();
        std::vector<int32_t> h_envnodes = g_envnodes.download_data();
        std::vector<int32_t> h_evtok = g_evtok.download_data();
        std::vector<uint64_t> h_tree = g_tree.download_data();
        std::vector<uint64_t> h_state = g_state.download_data();

        RpctHostInputsView hin = {};
        hin.op_kind = flat.op_kind.data();
        hin.op_arg_a = flat.op_arg_a.data();
        hin.op_arg_b = flat.op_arg_b.data();
        hin.op_token_offset = flat.op_token_offset.data();
        hin.op_token_len = flat.op_token_len.data();
        hin.op_tokens = flat.op_tokens.data();

        std::vector<RpctOpExpected> expected;
        oracle.step(run, hin, &expected);

        RpctHostOutputsView got = {};
        got.matched_prefix_len = h_matched.data();
        got.num_nodes = h_nnodes.data();
        got.num_tokens_cached = h_ntok.data();
        got.num_evicted_nodes = h_envnodes.data();
        got.evicted_tokens = h_evtok.data();
        got.tree_checksum = h_tree.data();
        got.state_checksum = h_state.data();

        ok = ok && rpct_check_all_outputs(expected, got, op_count, &error);

        if (results) {
            for (int i = 0; i < op_count; ++i) {
                OpResult r;
                r.matched_prefix_len = h_matched[i];
                r.num_nodes = h_nnodes[i];
                r.num_tokens_cached = h_ntok[i];
                r.num_evicted_nodes = h_envnodes[i];
                r.evicted_tokens = h_evtok[i];
                r.tree_checksum = h_tree[i];
                r.state_checksum = h_state[i];
                results->push_back(r);
            }
        }

        ++(*total);
        if (ok) {
            ++(*passed);
        } else {
            all_ok = false;
            if (first_error && first_error->empty()) {
                std::ostringstream oss;
                oss << sc.name << " step " << si << ": " << error;
                *first_error = oss.str();
            }
        }

        if (verbose) {
            std::printf("scenario %-22s step %02zu/%02zu ops=%2d %s%s%s\n",
                        sc.name.c_str(), si, sc.steps.size(), op_count,
                        ok ? "PASS" : "FAIL", ok ? "" : "  ", ok ? "" : error.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_results(const std::vector<OpResult>& a, const std::vector<OpResult>& b, std::string* error) {
    if (a.size() != b.size()) { if (error) *error = "op count mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].matched_prefix_len != b[i].matched_prefix_len ||
            a[i].num_nodes != b[i].num_nodes ||
            a[i].num_tokens_cached != b[i].num_tokens_cached ||
            a[i].num_evicted_nodes != b[i].num_evicted_nodes ||
            a[i].evicted_tokens != b[i].evicted_tokens ||
            a[i].tree_checksum != b[i].tree_checksum ||
            a[i].state_checksum != b[i].state_checksum) {
            if (error) { std::ostringstream o; o << "replay mismatch at op " << i; *error = o.str(); }
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
            std::vector<OpResult> base, replay;
            std::string error;
            const bool ok_base = run_scenario_once(sc, true, &base, &passed, &total, &error);
            const bool ok_replay = run_scenario_once(sc, false, &replay, &passed, &total, &error);

            if (ok_base && ok_replay) {
                std::string cerr;
                if (compare_results(base, replay, &cerr)) {
                    std::printf("scenario %-22s exact replay PASS\n", sc.name.c_str());
                } else {
                    all_ok = false;
                    std::printf("scenario %-22s exact replay FAIL  %s\n", sc.name.c_str(), cerr.c_str());
                }
            } else {
                all_ok = false;
                std::printf("scenario %-22s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return all_ok && passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
