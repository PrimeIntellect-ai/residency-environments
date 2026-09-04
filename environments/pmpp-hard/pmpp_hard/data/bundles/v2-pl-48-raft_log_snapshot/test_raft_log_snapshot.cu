// file: test_raft_log_snapshot.cu

#include "raft_log_snapshot_common.h"
#include "raft_log_snapshot_oracle.hpp"

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
            if (left[i] != kGuardByte) { if (error) { std::ostringstream o; o << "left guard corrupted for " << name << " at byte " << i; *error = o.str(); } return false; }
            if (right[i] != kGuardByte) { if (error) { std::ostringstream o; o << "right guard corrupted for " << name << " at byte " << i; *error = o.str(); } return false; }
        }
        return true;
    }
};

struct StepHost {
    RaftRunSpec run;
    std::vector<RaftOp> ops;
};

struct StepResult {
    RaftCounts counts{};
    uint64_t event_hash = 0;
    uint64_t log_hash = 0;
    uint64_t leader_state_hash = 0;
    uint64_t pending_rpc_hash = 0;
    uint64_t apply_hash = 0;
};

struct Scenario {
    std::string name;
    RaftProblemSpec spec;
    std::vector<StepHost> steps;
};

static RaftProblemSpec make_spec(int S, int Lcap, int P, int Ecap, int max_apply, int max_ops, int max_steps) {
    RaftProblemSpec spec = {};
    spec.abi_version = RAFT_ABI_VERSION;
    spec.server_count = S;
    spec.max_log_entries_per_server = Lcap;
    spec.max_pending_append_rpcs = P;
    spec.max_entries_per_append = Ecap;
    spec.max_apply_per_op = max_apply;
    spec.max_ops = max_ops;
    spec.max_steps = max_steps;
    spec.flags = 0;
    if (!raft_validate_problem_spec(&spec)) throw std::runtime_error("invalid RaftProblemSpec");
    return spec;
}

// op builders
static RaftOp op_become(int server, uint64_t term) { RaftOp o = {}; o.kind = RAFT_OP_BECOME_LEADER; o.i_a = server; o.u_a = term; return o; }
static RaftOp op_client(uint64_t cmd, int64_t payload) { RaftOp o = {}; o.kind = RAFT_OP_CLIENT_APPEND; o.u_a = cmd; o.value = payload; return o; }
static RaftOp op_send(int follower, int max_entries) { RaftOp o = {}; o.kind = RAFT_OP_SEND_APPEND; o.i_a = follower; o.i_b = max_entries; return o; }
static RaftOp op_deliver(uint64_t rpc_id) { RaftOp o = {}; o.kind = RAFT_OP_DELIVER_APPEND; o.u_a = rpc_id; return o; }
static RaftOp op_advance() { RaftOp o = {}; o.kind = RAFT_OP_ADVANCE_COMMIT; return o; }
static RaftOp op_apply(int server, int limit) { RaftOp o = {}; o.kind = RAFT_OP_APPLY; o.i_a = server; o.i_b = limit; return o; }
static RaftOp op_snapshot(int server) { RaftOp o = {}; o.kind = RAFT_OP_TAKE_SNAPSHOT; o.i_a = server; return o; }
static RaftOp op_install(int follower) { RaftOp o = {}; o.kind = RAFT_OP_INSTALL_SNAPSHOT; o.i_a = follower; return o; }

static StepHost make_step(const RaftProblemSpec& spec, int step_id, std::vector<RaftOp> ops) {
    StepHost s;
    s.run = {};
    s.run.abi_version = RAFT_ABI_VERSION;
    s.run.num_ops = (int32_t)ops.size();
    s.run.step_id = step_id;
    if (!raft_validate_run_spec(&s.run, &spec)) throw std::runtime_error("invalid RaftRunSpec");
    s.ops = std::move(ops);
    return s;
}

// ---------------------------------------------------------------------------
// Scenario 1: happy path replication + commit + apply + local snapshot.
//   3 servers, replicate to a majority, advance commit (current-term gate),
//   apply, take local snapshot, then continue.
// ---------------------------------------------------------------------------
static Scenario sc_happy_path() {
    Scenario sc;
    sc.name = "happy_path_replicate_commit_snapshot";
    sc.spec = make_spec(3, 32, 8, 8, 32, 32, 24);

    // step 0: elect leader 0 at term 1, append 3 client entries
    sc.steps.push_back(make_step(sc.spec, 0, {
        op_become(0, 1), op_client(100, 11), op_client(101, 22), op_client(102, 33),
    }));
    // step 1: send to followers 1 and 2 (rpc 1 and 2), deliver both
    sc.steps.push_back(make_step(sc.spec, 1, {
        op_send(1, 8), op_send(2, 8), op_deliver(1), op_deliver(2),
    }));
    // step 2: advance commit (entries are current term => should advance to 3)
    sc.steps.push_back(make_step(sc.spec, 2, { op_advance() }));
    // step 3: apply 2 on leader, then 1 more
    sc.steps.push_back(make_step(sc.spec, 3, { op_apply(0, 2), op_apply(0, 1) }));
    // step 4: apply on followers
    sc.steps.push_back(make_step(sc.spec, 4, { op_apply(1, 10), op_apply(2, 10) }));
    // step 5: take snapshot on leader (last_applied=3 > snap_index 0)
    sc.steps.push_back(make_step(sc.spec, 5, { op_snapshot(0) }));
    // step 6: snapshot on follower 1 too
    sc.steps.push_back(make_step(sc.spec, 6, { op_snapshot(1) }));
    // step 7: more client appends after snapshot
    sc.steps.push_back(make_step(sc.spec, 7, {
        op_client(103, 44), op_client(104, 55), op_send(2, 8), op_deliver(3),
    }));
    // step 8: empty step (no ops)
    sc.steps.push_back(make_step(sc.spec, 8, {}));
    // step 9: advance + apply leader
    sc.steps.push_back(make_step(sc.spec, 9, { op_advance(), op_apply(0, 10) }));
    return sc;
}

// ---------------------------------------------------------------------------
// Scenario 2: stale RPC delivery after leader change.
//   Leader 0 sends an RPC, then leader changes to server 1 at higher term,
//   delivering the old RPC must be APPEND_STALE (leader no longer current /
//   term mismatch).
// ---------------------------------------------------------------------------
static Scenario sc_stale_rpc() {
    Scenario sc;
    sc.name = "stale_rpc_after_leader_change";
    sc.spec = make_spec(5, 32, 16, 8, 32, 32, 16);

    sc.steps.push_back(make_step(sc.spec, 0, {
        op_become(0, 2), op_client(1, 10), op_client(2, 20), op_client(3, 30),
    }));
    // leader 0 sends two RPCs (rpc 1 -> follower 1, rpc 2 -> follower 2) but
    // does NOT deliver yet
    sc.steps.push_back(make_step(sc.spec, 1, { op_send(1, 8), op_send(2, 8) }));
    // leader changes to server 1 at term 3 (server 0 becomes follower)
    sc.steps.push_back(make_step(sc.spec, 2, { op_become(1, 3) }));
    // delivering rpc 1 and 2 => stale (R.leader 0 != current leader 1)
    sc.steps.push_back(make_step(sc.spec, 3, { op_deliver(1), op_deliver(2) }));
    // new leader appends and replicates fresh
    sc.steps.push_back(make_step(sc.spec, 4, {
        op_client(4, 40), op_send(0, 8), op_send(2, 8), op_deliver(3), op_deliver(4),
    }));
    // deliver a non-existent rpc => invalid
    sc.steps.push_back(make_step(sc.spec, 5, { op_deliver(999) }));
    sc.steps.push_back(make_step(sc.spec, 6, { op_advance(), op_apply(1, 10) }));
    return sc;
}

// ---------------------------------------------------------------------------
// Scenario 3: conflict backtracking — both conflict_term and conflict_index
//   paths. We give a follower a divergent log via a prior term's leader, then
//   a new leader forces conflict resolution + suffix deletion.
// ---------------------------------------------------------------------------
static Scenario sc_conflict_backtrack() {
    Scenario sc;
    sc.name = "conflict_backtrack_and_suffix_delete";
    sc.spec = make_spec(3, 32, 16, 8, 32, 32, 20);

    // term 1: leader 0 builds log [1,2,3] and replicates to follower 1 fully,
    // follower 2 partially.
    sc.steps.push_back(make_step(sc.spec, 0, {
        op_become(0, 1), op_client(10, 1), op_client(11, 2), op_client(12, 3),
        op_send(1, 8), op_deliver(1),
    }));
    // follower 2 gets only first entry via capped send (max_entries=1)
    sc.steps.push_back(make_step(sc.spec, 1, { op_send(2, 1), op_deliver(2) }));
    // term 2: leader 1 takes over. It has full log [1,2,3]. Append new entries.
    sc.steps.push_back(make_step(sc.spec, 2, {
        op_become(1, 2), op_client(20, 7), op_client(21, 8),
    }));
    // leader 1 sends to follower 2 with default next_index (lli+1=6), which is
    // > follower2 lli => conflict (prev absent) => conflict_index backtrack.
    sc.steps.push_back(make_step(sc.spec, 3, { op_send(2, 8), op_deliver(3) }));
    // After backtrack next_index decreased; send again to make progress.
    sc.steps.push_back(make_step(sc.spec, 4, { op_send(2, 8), op_deliver(4) }));
    sc.steps.push_back(make_step(sc.spec, 5, { op_send(2, 8), op_deliver(5) }));
    sc.steps.push_back(make_step(sc.spec, 6, { op_send(2, 8), op_deliver(6) }));
    // send to follower 0 (old leader). 0 has [1,2,3] term1; leader1 log has
    // [1,2,3 term1, 4,5 term2]. Should reconcile without deletion.
    sc.steps.push_back(make_step(sc.spec, 7, { op_send(0, 8), op_deliver(7) }));
    sc.steps.push_back(make_step(sc.spec, 8, { op_advance(), op_apply(1, 10) }));
    return sc;
}

// ---------------------------------------------------------------------------
// Scenario 4: APPEND_NEEDS_SNAPSHOT + INSTALL_SNAPSHOT with retained suffix
//   and with full wipe.
// ---------------------------------------------------------------------------
static Scenario sc_install_snapshot() {
    Scenario sc;
    sc.name = "needs_snapshot_then_install";
    sc.spec = make_spec(3, 32, 16, 8, 32, 32, 24);

    // leader 0 builds and commits/applies several entries, snapshots up to 4.
    sc.steps.push_back(make_step(sc.spec, 0, {
        op_become(0, 1), op_client(1, 10), op_client(2, 20), op_client(3, 30),
        op_client(4, 40), op_client(5, 50),
    }));
    sc.steps.push_back(make_step(sc.spec, 1, {
        op_send(1, 8), op_deliver(1), op_send(2, 8), op_deliver(2), op_advance(),
    }));
    sc.steps.push_back(make_step(sc.spec, 2, { op_apply(0, 10) }));
    sc.steps.push_back(make_step(sc.spec, 3, { op_snapshot(0) }));  // snap_index=5
    // Now reset follower 2's next_index by forcing it behind: take another
    // leader snapshot is enough. Send to follower 2: next_index already 6
    // (it received up to 5), so set it behind by simulating a fresh follower.
    // Instead, demonstrate NEEDS_SNAPSHOT on a follower that never caught up:
    // re-elect to bump and create a follower far behind via conflict.
    // Simpler: elect leader and have follower 2 lagging by resetting via a new
    // leader that has snapshot. We make leader 0 snapshot beyond follower-2's
    // next_index using a manual SEND when next_index <= snapshot_index.
    // Build a 4th client entry then snapshot more so follower-2 needs snapshot.
    sc.steps.push_back(make_step(sc.spec, 4, {
        op_client(6, 60), op_client(7, 70),
    }));
    // follower 2 currently has up to index 5 (next_index=6). Leader snap_index
    // is 5, so 6 > 5, normal send works. To trigger NEEDS_SNAPSHOT we need a
    // follower whose next_index <= leader snap_index. Re-elect server 1 to
    // wipe leader-volatile match/next: become_leader sets next_index for all
    // followers to lli+1, and a fresh snapshot makes any *manually decreased*
    // next_index trip the guard. Force next_index of follower 2 down via a
    // term-reject path is complex; instead install snapshot directly.
    sc.steps.push_back(make_step(sc.spec, 5, { op_install(1) }));   // install onto follower1 (retain or wipe)
    sc.steps.push_back(make_step(sc.spec, 6, { op_install(2) }));   // install onto follower2
    sc.steps.push_back(make_step(sc.spec, 7, { op_install(1) }));   // install again => NOOP (snap_index equal)
    // follower applies from installed snapshot baseline
    sc.steps.push_back(make_step(sc.spec, 8, { op_apply(1, 10), op_apply(2, 10) }));
    sc.steps.push_back(make_step(sc.spec, 9, { op_advance(), op_apply(0, 10) }));
    return sc;
}

// ---------------------------------------------------------------------------
// Scenario 5: NEEDS_SNAPSHOT directly. Build leader snapshot, then re-elect a
//   different server whose log lags, send while its next_index sits at/under
//   leader snapshot index.
// ---------------------------------------------------------------------------
static Scenario sc_needs_snapshot() {
    Scenario sc;
    sc.name = "append_needs_snapshot_path";
    sc.spec = make_spec(3, 32, 16, 8, 32, 32, 20);

    // leader 0: append 4, replicate to follower1 only, commit via majority?
    // majority of 3 = 2 (leader + f1). apply + snapshot leader to index 4.
    sc.steps.push_back(make_step(sc.spec, 0, {
        op_become(0, 1), op_client(1, 1), op_client(2, 2), op_client(3, 3), op_client(4, 4),
        op_send(1, 8), op_deliver(1), op_advance(), op_apply(0, 10),
    }));
    sc.steps.push_back(make_step(sc.spec, 1, { op_snapshot(0) }));  // leader snap_index=4
    // follower 2 was never sent anything; its next_index = 5 (set at election
    // = lli+1 = 0+1 = 1 actually). At election lli(leader)=0 so next_index=1.
    // After appends, next_index[2] still 1 (never updated). leader snap_index=4
    // => 1 <= 4 => NEEDS_SNAPSHOT.
    sc.steps.push_back(make_step(sc.spec, 2, { op_send(2, 8) }));   // APPEND_NEEDS_SNAPSHOT
    // install snapshot on follower 2 (full wipe, follower2 empty log)
    sc.steps.push_back(make_step(sc.spec, 3, { op_install(2) }));
    // now next_index[2] = snap_index+1 = 5, send works
    sc.steps.push_back(make_step(sc.spec, 4, { op_client(5, 5), op_send(2, 8), op_deliver(2) }));
    sc.steps.push_back(make_step(sc.spec, 5, { op_advance(), op_apply(0, 10), op_apply(2, 10) }));
    return sc;
}

// ---------------------------------------------------------------------------
// Scenario 6: small cluster (server_count=1) edge + invalids + log-full +
//   apply-empty + commit-noop + snapshot invalid.
// ---------------------------------------------------------------------------
static Scenario sc_edges() {
    Scenario sc;
    sc.name = "edges_invalids_logfull";
    sc.spec = make_spec(1, 4, 4, 4, 4, 32, 16);  // single server, tiny log cap

    // no leader yet: client append invalid, advance invalid, send invalid
    sc.steps.push_back(make_step(sc.spec, 0, {
        op_client(1, 1), op_advance(), op_send(0, 4), op_apply(0, 4),
    }));
    // become leader (server 0), majority=1 so it can self-commit
    sc.steps.push_back(make_step(sc.spec, 1, {
        op_become(0, 1), op_client(10, 1), op_client(11, 2), op_client(12, 3), op_client(13, 4),
    }));
    // log now full (cap 4): another client => REJECT_LOG_FULL
    sc.steps.push_back(make_step(sc.spec, 2, { op_client(14, 5) }));
    // advance: single-server majority => commit to last; apply all
    sc.steps.push_back(make_step(sc.spec, 3, { op_advance(), op_apply(0, 10) }));
    // apply again => apply_empty (nothing left)
    sc.steps.push_back(make_step(sc.spec, 4, { op_apply(0, 4) }));
    // advance again => commit_noop
    sc.steps.push_back(make_step(sc.spec, 5, { op_advance() }));
    // snapshot (last_applied=4 > snap_index 0) ok; then snapshot again invalid
    sc.steps.push_back(make_step(sc.spec, 6, { op_snapshot(0), op_snapshot(0) }));
    // invalid become_leader: term < current_term
    sc.steps.push_back(make_step(sc.spec, 7, { op_become(0, 0) }));
    // out of range server ops
    sc.steps.push_back(make_step(sc.spec, 8, { op_apply(5, 1), op_snapshot(7), op_send(3, 2), op_install(2) }));
    // apply with limit 0 => valid no-op (no event, no invalid)
    sc.steps.push_back(make_step(sc.spec, 9, { op_apply(0, 0) }));
    return sc;
}

// ---------------------------------------------------------------------------
// Scenario 7: the three remaining hard paths —
//   APPEND_SEND_REJECT (pending table full),
//   APPEND_TERM_REJECT (follower term > stale-but-current leader term),
//   APPEND_FOLLOWER_OOM (success path would exceed follower log capacity).
// ---------------------------------------------------------------------------
static Scenario sc_reject_paths() {
    Scenario sc;
    sc.name = "send_reject_term_reject_oom";
    // tiny pending table (P=2) and tiny log cap (6) to force the limits.
    sc.spec = make_spec(3, 6, 2, 6, 16, 32, 20);

    // --- SEND_REJECT: leader 0 term 5, append entries, issue 3 sends without
    // delivering. Pending cap = 2, so the 3rd SEND => APPEND_SEND_REJECT.
    sc.steps.push_back(make_step(sc.spec, 0, {
        op_become(0, 5), op_client(1, 1), op_client(2, 2),
        op_send(1, 4), op_send(2, 4), op_send(1, 4),  // 3rd => reject
    }));
    // clear pending by delivering the two live ones (rpc 1->f1, rpc 2->f2)
    sc.steps.push_back(make_step(sc.spec, 1, { op_deliver(1), op_deliver(2) }));

    // --- TERM_REJECT setup. Leader 0 term 5 sends rpc to follower 2 (rpc 3).
    sc.steps.push_back(make_step(sc.spec, 2, { op_send(2, 4) }));
    // Elect server 1 leader at term 7, replicate to follower 2 (raises term 2->7).
    sc.steps.push_back(make_step(sc.spec, 3, {
        op_become(1, 7), op_client(3, 3), op_send(2, 4), op_deliver(4),
    }));
    // Re-elect server 0 at term 5 (still valid: 5 >= current_term[0]=5).
    sc.steps.push_back(make_step(sc.spec, 4, { op_become(0, 5) }));
    // Deliver the stale rpc 3: leader 0 is current at term 5 (== R.leader_term),
    // but follower 2 current_term=7 > 5 => APPEND_TERM_REJECT.
    sc.steps.push_back(make_step(sc.spec, 5, { op_deliver(3) }));

    // --- OOM. Build leader 1 (term 8) up to a full log [1..6] (cap=6), apply,
    // and snapshot so leader can advance beyond cap. Follower 0 holds a full
    // log [1..6] too; then leader (post-snapshot) sends entries at higher
    // indices whose append would push follower 0 above its capacity.
    sc.steps.push_back(make_step(sc.spec, 6, {
        op_become(1, 8), op_client(4, 4), op_client(5, 5), op_client(6, 6),
    }));
    // leader 1 log now [1(t5),2(t5),3(t7),4(t8),5(t8),6(t8)] = 6 entries (cap).
    // Replicate fully to follower 0 (two capped sends of 4 then remainder).
    sc.steps.push_back(make_step(sc.spec, 7, { op_send(0, 4), op_deliver(5) }));
    sc.steps.push_back(make_step(sc.spec, 8, { op_send(0, 4), op_deliver(6) }));
    // follower 0 now [1..6] (full). Commit + apply leader to 6, snapshot leader
    // up to 4 so leader log becomes [5,6] and it can append [7,8] within cap.
    sc.steps.push_back(make_step(sc.spec, 9, {
        op_send(2, 4), op_deliver(7), op_advance(), op_apply(1, 10),
    }));
    sc.steps.push_back(make_step(sc.spec, 10, { op_snapshot(1) }));  // leader snap_index=6, log empty
    // append new entries 7,8 (leader log [7,8]); follower 0 still full [1..6]
    // with next_index 7 (prev_index=6 == leader nothing? prev_index 6 == snap?).
    sc.steps.push_back(make_step(sc.spec, 11, { op_client(7, 7), op_client(8, 8) }));
    // send to follower 0: next_index[0]=7, prev_index=6==leader snap_index =>
    // prev_term=snap_term, consistent at boundary. entries [7,8]. follower 0
    // has 6 entries; appending 2 => 8 > cap 6 => APPEND_FOLLOWER_OOM.
    sc.steps.push_back(make_step(sc.spec, 12, { op_send(0, 4), op_deliver(8) }));
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(sc_happy_path());
    v.push_back(sc_stale_rpc());
    v.push_back(sc_conflict_backtrack());
    v.push_back(sc_install_snapshot());
    v.push_back(sc_needs_snapshot());
    v.push_back(sc_edges());
    v.push_back(sc_reject_paths());
    return v;
}

static bool run_one_step(
    const RaftProblemSpec& spec, const StepHost& step, RaftOracleState* oracle,
    void* state, void* workspace, size_t workspace_bytes, cudaStream_t stream,
    StepResult* result, std::string* error) {
    DeviceBuffer<RaftOp> d_ops;
    d_ops.allocate(std::max<size_t>(1, step.ops.size()));
    {
        std::vector<RaftOp> host = step.ops;
        if (host.empty()) { RaftOp z = {}; z.kind = -1; host.push_back(z); }
        d_ops.upload(host);
    }

    GuardedDeviceBuffer<RaftCounts> d_counts;
    GuardedDeviceBuffer<uint64_t> d_event, d_log, d_leader, d_pending, d_apply;
    d_counts.allocate(1);
    d_event.allocate(1); d_log.allocate(1); d_leader.allocate(1);
    d_pending.allocate(1); d_apply.allocate(1);

    RaftInputs inputs = {};
    inputs.ops = d_ops.ptr;

    RaftOutputs outputs = {};
    outputs.counts = d_counts.ptr;
    outputs.raft_event_hash = d_event.ptr;
    outputs.log_hash = d_log.ptr;
    outputs.leader_state_hash = d_leader.ptr;
    outputs.pending_rpc_hash = d_pending.ptr;
    outputs.apply_hash = d_apply.ptr;

    // snapshot the input bytes to verify immutability afterward
    const std::vector<RaftOp> before = d_ops.download();

    CUDA_CHECK(solution_run(state, &step.run, &inputs, &outputs, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // input immutability
    const std::vector<RaftOp> after = d_ops.download();
    if (std::memcmp(before.data(), after.data(), sizeof(RaftOp) * before.size()) != 0) {
        if (error) *error = "input ops buffer modified by solution";
        return false;
    }

    if (!d_counts.check_guards("counts", error)) return false;
    if (!d_event.check_guards("raft_event_hash", error)) return false;
    if (!d_log.check_guards("log_hash", error)) return false;
    if (!d_leader.check_guards("leader_state_hash", error)) return false;
    if (!d_pending.check_guards("pending_rpc_hash", error)) return false;
    if (!d_apply.check_guards("apply_hash", error)) return false;

    const std::vector<RaftCounts> h_counts = d_counts.download_data();
    const std::vector<uint64_t> h_event = d_event.download_data();
    const std::vector<uint64_t> h_log = d_log.download_data();
    const std::vector<uint64_t> h_leader = d_leader.download_data();
    const std::vector<uint64_t> h_pending = d_pending.download_data();
    const std::vector<uint64_t> h_apply = d_apply.download_data();

    RaftExpected expected;
    oracle->step_once(step.run, step.ops.data(), &expected);

    RaftHostOutputsView got = {};
    got.counts = h_counts.data();
    got.raft_event_hash = h_event.data();
    got.log_hash = h_log.data();
    got.leader_state_hash = h_leader.data();
    got.pending_rpc_hash = h_pending.data();
    got.apply_hash = h_apply.data();

    if (!raft_check_all_outputs(expected, got, error)) return false;

    if (result) {
        result->counts = h_counts[0];
        result->event_hash = h_event[0];
        result->log_hash = h_log[0];
        result->leader_state_hash = h_leader[0];
        result->pending_rpc_hash = h_pending[0];
        result->apply_hash = h_apply[0];
    }
    return true;
}

static bool run_scenario_once(
    const Scenario& sc, bool verbose, std::vector<StepResult>* results,
    int* passed_steps, int* total_steps, std::string* first_error) {
    size_t workspace_bytes = solution_workspace_bytes(&sc.spec);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    RaftOracleState oracle;
    oracle.init(sc.spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) { results->clear(); results->reserve(sc.steps.size()); }

    bool all_ok = true;
    for (size_t i = 0; i < sc.steps.size(); ++i) {
        StepResult result;
        std::string error;
        const bool ok = run_one_step(sc.spec, sc.steps[i], &oracle, state,
                                     workspace.ptr, workspace_bytes, stream,
                                     results ? &result : nullptr, &error);
        ++(*total_steps);
        if (ok) ++(*passed_steps);
        else {
            all_ok = false;
            if (first_error && first_error->empty()) {
                std::ostringstream oss; oss << sc.name << " step " << i << ": " << error; *first_error = oss.str();
            }
        }
        if (results) results->push_back(result);
        if (verbose) {
            std::printf("scenario %-40s step %02zu/%02zu ops=%d %s%s%s\n",
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
        if (std::memcmp(&a[i].counts, &b[i].counts, sizeof(RaftCounts)) != 0 ||
            a[i].event_hash != b[i].event_hash || a[i].log_hash != b[i].log_hash ||
            a[i].leader_state_hash != b[i].leader_state_hash ||
            a[i].pending_rpc_hash != b[i].pending_rpc_hash || a[i].apply_hash != b[i].apply_hash) {
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

        for (const Scenario& sc : scenarios) {
            std::vector<StepResult> base_results, replay_results;
            std::string error;

            const bool ok_base = run_scenario_once(sc, true, &base_results, &passed, &total, &error);
            const bool ok_replay = run_scenario_once(sc, false, &replay_results, &passed, &total, &error);

            if (ok_base && ok_replay) {
                std::string cmp;
                if (compare_results(base_results, replay_results, &cmp))
                    std::printf("scenario %-40s exact replay PASS\n", sc.name.c_str());
                else { all_ok = false; std::printf("scenario %-40s exact replay FAIL  %s\n", sc.name.c_str(), cmp.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-40s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return all_ok && passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
