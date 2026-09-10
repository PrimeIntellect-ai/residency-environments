// file: test_incremental_gc_weakref.cu
//
// 3-way validation harness for T53. Builds deeply adversarial op-stream
// scenarios (ephemeron fixpoint chains, finalizer resurrection, weak-ref
// nulling across increments, remembered-set staleness, promotion), runs the
// solution under test against the host oracle for every op, with guard
// buffers, input-immutability (no inputs here, so op-stream immutability),
// and exact-replay determinism checks.

#include "incremental_gc_weakref_common.h"
#include "incremental_gc_weakref_oracle.hpp"

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
    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count > 0) CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        return host;
    }
    bool check_guards(const char* name, std::string* err) const {
        std::vector<uint8_t> left(kGuardBytes), right(kGuardBytes);
        CUDA_CHECK(cudaMemcpy(left.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(right.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (left[i] != kGuardByte) { if (err){ std::ostringstream o; o<<"left guard "<<name<<" byte "<<i; *err=o.str(); } return false; }
            if (right[i] != kGuardByte){ if (err){ std::ostringstream o; o<<"right guard "<<name<<" byte "<<i; *err=o.str(); } return false; }
        }
        return true;
    }
};

// ----- scenario builder helpers -----
struct Scenario {
    std::string name;
    IgcwProblemSpec spec;
    std::vector<IgcwRunSpec> ops;
};

struct OpSnapshot {
    int32_t counts[IGCW_NUM_COUNTS];
    uint64_t eh, hh, rh, eph, rem, ch;
    int32_t inv;
};

static IgcwRunSpec mk(int opcode, int op_index, int a0=0,int a1=0,int a2=0,int a3=0,int64_t size=0){
    IgcwRunSpec r; std::memset(&r,0,sizeof(r));
    r.abi_version=IGCW_ABI_VERSION; r.opcode=opcode; r.op_index=op_index;
    r.a0=a0; r.a1=a1; r.a2=a2; r.a3=a3; r.size_arg=size;
    return r;
}

// Convenience push that auto-increments op_index.
struct OpBuilder {
    std::vector<IgcwRunSpec>& ops;
    int idx = 0;
    explicit OpBuilder(std::vector<IgcwRunSpec>& o):ops(o){}
    void alloc(int gen,int tag,int slot,int64_t size){ ops.push_back(mk(IGCW_OP_ALLOC,idx++,gen,tag,slot,0,size)); }
    void set_root(int rid,int obj){ ops.push_back(mk(IGCW_OP_SET_ROOT,idx++,rid,obj)); }
    void clear_root(int rid){ ops.push_back(mk(IGCW_OP_CLEAR_ROOT,idx++,rid)); }
    void set_strong(int src,int slot,int dst){ ops.push_back(mk(IGCW_OP_SET_STRONG,idx++,src,slot,dst)); }
    void clear_strong(int src,int slot){ ops.push_back(mk(IGCW_OP_CLEAR_STRONG,idx++,src,slot)); }
    void set_weak(int src,int slot,int dst){ ops.push_back(mk(IGCW_OP_SET_WEAK,idx++,src,slot,dst)); }
    void set_eph(int eid,int owner,int key,int val){ ops.push_back(mk(IGCW_OP_SET_EPHEMERON,idx++,eid,owner,key,val)); }
    void del_eph(int eid){ ops.push_back(mk(IGCW_OP_DELETE_EPHEMERON,idx++,eid)); }
    void start_minor(){ ops.push_back(mk(IGCW_OP_START_MINOR,idx++)); }
    void start_full(){ ops.push_back(mk(IGCW_OP_START_FULL,idx++)); }
    void gc_step(int mb,int sb){ ops.push_back(mk(IGCW_OP_GC_STEP,idx++,mb,sb)); }
    void run_fin(int limit){ ops.push_back(mk(IGCW_OP_RUN_FINALIZERS,idx++,limit)); }
    // run an incremental full GC to completion with small budgets (forces
    // many phase transitions / pauses to exercise persistence).
    void full_gc_incremental(int steps){ start_full(); for(int i=0;i<steps;++i) gc_step(1,1); }
    void minor_gc_incremental(int steps){ start_minor(); for(int i=0;i<steps;++i) gc_step(1,1); }
};

static IgcwProblemSpec mk_spec(int MO,int RC,int SS,int WS,int ME,int MQ,int FQ,int YST){
    IgcwProblemSpec s; std::memset(&s,0,sizeof(s));
    s.abi_version=IGCW_ABI_VERSION; s.max_objects=MO; s.root_count=RC;
    s.strong_slots_per_object=SS; s.weak_slots_per_object=WS;
    s.max_ephemerons=ME; s.max_mark_queue=MQ; s.max_finalizer_queue=FQ; s.young_survive_threshold=YST;
    return s;
}

// Scenario 1: basic alloc + full GC sweeps garbage, keeps reachable.
static Scenario sc_basic_full() {
    Scenario sc; sc.name="basic_full_sweep";
    sc.spec=mk_spec(32,8,4,2,16,256,64,2);
    OpBuilder b(sc.ops);
    // objects 1..6
    for(int i=0;i<6;++i) b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,100+i);
    b.set_root(0,1); b.set_root(1,2);
    b.set_strong(1,0,3); b.set_strong(3,0,4); // 1->3->4 reachable
    // 2 reachable; 5,6 garbage
    b.full_gc_incremental(40);
    // After GC: 5 and 6 freed, ids reused on next alloc (lowest free).
    b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,777);
    return sc;
}

// Scenario 2: weak references nulled on collection; survive when target kept.
static Scenario sc_weak_nulling() {
    Scenario sc; sc.name="weak_nulling";
    sc.spec=mk_spec(32,8,4,3,16,256,64,2);
    OpBuilder b(sc.ops);
    for(int i=0;i<5;++i) b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,10+i);
    b.set_root(0,1);
    b.set_strong(1,0,2);      // 2 kept
    b.set_weak(1,0,2);        // weak to kept -> survives
    b.set_weak(1,1,3);        // weak to garbage 3 -> nulled
    b.set_weak(2,0,4);        // weak to garbage 4 -> nulled
    // 3,4,5 are garbage
    b.full_gc_incremental(40);
    return sc;
}

// Scenario 3: ephemeron fixpoint chain. value reachable only if key reachable.
// Build chain: root->k1 ; eph0(owner=o, key=k1, value=k2); eph1(key=k2,value=k3)
// So k2 reachable via eph0 only after k1 marked; k3 via eph1 after k2 -> fixpoint.
static Scenario sc_ephemeron_chain() {
    Scenario sc; sc.name="ephemeron_fixpoint_chain";
    sc.spec=mk_spec(32,8,4,2,16,256,64,2);
    OpBuilder b(sc.ops);
    // ids: 1=owner, 2=k1, 3=k2, 4=k3, 5=k4(dead key), 6=v_dead
    for(int i=0;i<6;++i) b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,1+i);
    b.set_root(0,1);          // owner reachable
    b.set_root(1,2);          // k1 reachable
    b.set_eph(0,1,2,3);       // key=k1(reachable)->value=k2 marked
    b.set_eph(1,1,3,4);       // key=k2 -> value=k3 (needs k2 first => fixpoint)
    b.set_eph(2,1,5,6);       // key=k4(dead)->value=6 : stays white, cleared
    b.full_gc_incremental(60);
    return sc;
}

// Scenario 4: finalizer resurrection. White finalizable obj enqueued, resurrected
// into its root slot, never freed this cycle; not finalized twice next cycle.
static Scenario sc_finalizer_resurrect() {
    Scenario sc; sc.name="finalizer_resurrect";
    sc.spec=mk_spec(32,8,4,2,16,256,64,2);
    OpBuilder b(sc.ops);
    // 1 root-held; 2 has finalizer with root slot 3; 3 has finalizer no slot (UINT32_MAX)
    b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,1);   // id1
    b.alloc(IGCW_GEN_YOUNG,7,3,2);              // id2 finalizer tag7 resurrect into root3
    b.alloc(IGCW_GEN_YOUNG,9,(int)IGCW_U32_MAX,3); // id3 finalizer tag9 no resurrect slot
    b.set_root(0,1);
    // 2 and 3 unreachable but finalizable -> enqueued in finalize_scan
    b.full_gc_incremental(60);
    // Now run finalizers: 2 resurrected into root3, 3 just runs.
    b.run_fin(8);
    // Next full GC: object 2 now reachable via root3, object 3 should be swept,
    // and 3 must NOT be re-enqueued (already finalized).
    b.full_gc_incremental(60);
    b.run_fin(8);
    return sc;
}

// Scenario 5: minor GC + remembered set + promotion + staleness.
static Scenario sc_minor_remembered_promote() {
    Scenario sc; sc.name="minor_remembered_promote";
    sc.spec=mk_spec(40,8,4,2,16,256,64,1); // YST=1 -> promote after one survived minor
    OpBuilder b(sc.ops);
    // old object 1, young objects 2,3,4
    b.alloc(IGCW_GEN_OLD,0,IGCW_U32_MAX,1);   // id1 OLD
    b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,2); // id2
    b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,3); // id3
    b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,4); // id4
    b.set_root(0,1);
    b.set_strong(1,0,2);  // OLD->YOUNG remembered (1,0)->2
    b.set_strong(1,1,3);  // OLD->YOUNG remembered (1,1)->3
    // 4 is garbage young
    b.minor_gc_incremental(40); // 2,3 kept+promoted (YST=1), 4 freed
    // Now clear strong (1,1) -> remembered entry (1,1) becomes stale, not removed now
    b.clear_strong(1,1);
    // allocate young 5 (id reused = 4)
    b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,5);
    b.set_strong(1,2,5); // OLD(1)->YOUNG(5) remembered (1,2)
    b.minor_gc_incremental(40); // remembered scan drops stale (1,1); keeps reachable via (1,0)?,(1,2)
    return sc;
}

// Scenario 6: barriers during incremental GC. Mutate roots/strong mid-cycle.
static Scenario sc_incremental_barriers() {
    Scenario sc; sc.name="incremental_barriers";
    sc.spec=mk_spec(40,8,4,2,16,256,64,2);
    OpBuilder b(sc.ops);
    for(int i=0;i<8;++i) b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,i+1);
    b.set_root(0,1);
    b.set_strong(1,0,2);
    b.start_full();
    b.gc_step(1,1); // process some marking (mark obj1 -> grey 2)
    // mid-cycle: write barrier. object1 is BLACK now; set strong 1->5 should mark 5.
    b.set_strong(1,1,5);
    // root write barrier: set_root to white obj 7 -> marks 7
    b.set_root(2,7);
    // allocate during gc -> black object
    b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,999);
    // finish GC
    for(int i=0;i<40;++i) b.gc_step(2,2);
    return sc;
}

// Scenario 7: ephemeron whose key is collected -> value never marked, both
// the value (if otherwise white) cleared, plus owner-free cascade on sweep.
static Scenario sc_ephemeron_clear_owner_free() {
    Scenario sc; sc.name="ephemeron_clear_owner_free";
    sc.spec=mk_spec(32,8,4,2,16,256,64,2);
    OpBuilder b(sc.ops);
    for(int i=0;i<6;++i) b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,i+1);
    b.set_root(0,1);                 // only 1 reachable
    b.set_eph(0,1,2,3);              // owner=1 key=2(dead) value=3(dead)
    b.set_eph(1,2,4,5);              // owner=2(dead) -> on sweep of 2, owner-free
    b.full_gc_incremental(60);
    return sc;
}

// Scenario 8: invalid ops + OOM coverage.
static Scenario sc_invalid_and_oom() {
    Scenario sc; sc.name="invalid_and_oom";
    sc.spec=mk_spec(4,4,2,1,2,64,4,2);
    OpBuilder b(sc.ops);
    b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,1);
    b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,2);
    b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,3);
    b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,4);
    b.alloc(IGCW_GEN_YOUNG,0,IGCW_U32_MAX,5);  // OOM (max_objects=4)
    b.alloc(5,0,IGCW_U32_MAX,6);               // invalid generation
    b.set_root(99,1);                          // invalid root id
    b.set_strong(1,9,2);                       // invalid slot
    b.set_strong(77,0,2);                      // invalid src
    b.set_eph(0,1,2,3); b.set_eph(1,1,2,3); b.set_eph(5,1,2,3); // 3rd: id out of range invalid (ME=2 -> ids 0,1)
    b.del_eph(7);                              // absent
    b.gc_step(1,1);                            // idle -> invalid
    b.run_fin(0);                              // valid no-op
    b.set_eph(0,1,2,3); // valid update
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(sc_basic_full());
    v.push_back(sc_weak_nulling());
    v.push_back(sc_ephemeron_chain());
    v.push_back(sc_finalizer_resurrect());
    v.push_back(sc_minor_remembered_promote());
    v.push_back(sc_incremental_barriers());
    v.push_back(sc_ephemeron_clear_owner_free());
    v.push_back(sc_invalid_and_oom());
    return v;
}

// Run one scenario once. Returns success; fills per-op snapshots for replay.
static bool run_once(const Scenario& sc, bool verbose,
                     std::vector<OpSnapshot>* snaps,
                     int* passed, int* total, std::string* first_err) {
    IgcwProblemSpec spec = sc.spec;

    size_t workspace_bytes = solution_workspace_bytes(&spec);
    // MANDATE: clamp only; never fail-guard on workspace_bytes == 0.

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Workspace buffer (clamp-only).
    uint8_t* d_workspace = nullptr;
    size_t alloc_ws = std::max<size_t>(workspace_bytes, 1);
    CUDA_CHECK(cudaMalloc((void**)&d_workspace, alloc_ws));

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    IgcwOracle oracle;
    oracle.init(spec);

    if (snaps) { snaps->clear(); snaps->reserve(sc.ops.size()); }

    bool all_ok = true;

    for (size_t i = 0; i < sc.ops.size(); ++i) {
        const IgcwRunSpec op = sc.ops[i];
        // Copy op to a buffer so we can verify it is not mutated by the solution.
        IgcwRunSpec op_copy = op;

        GuardedDeviceBuffer<int32_t> d_counts;
        GuardedDeviceBuffer<uint64_t> d_eh, d_hh, d_rh, d_eph, d_rem, d_ch;
        GuardedDeviceBuffer<int32_t> d_inv;
        d_counts.allocate(IGCW_NUM_COUNTS);
        d_eh.allocate(1); d_hh.allocate(1); d_rh.allocate(1);
        d_eph.allocate(1); d_rem.allocate(1); d_ch.allocate(1);
        d_inv.allocate(1);

        IgcwOutputs outputs; std::memset(&outputs,0,sizeof(outputs));
        outputs.counts = d_counts.ptr;
        outputs.gc_event_hash = d_eh.ptr;
        outputs.heap_hash = d_hh.ptr;
        outputs.root_hash = d_rh.ptr;
        outputs.ephemeron_hash = d_eph.ptr;
        outputs.remembered_hash = d_rem.ptr;
        outputs.gc_controller_hash = d_ch.ptr;
        outputs.invalid_flag = d_inv.ptr;

        std::string err;

        CUDA_CHECK(solution_run(state, &op_copy, nullptr, &outputs, d_workspace, alloc_ws, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        bool ok = true;
        // op-stream immutability
        if (std::memcmp(&op_copy, &op, sizeof(IgcwRunSpec)) != 0) { ok=false; err="run spec mutated"; }
        ok = ok && d_counts.check_guards("counts",&err);
        ok = ok && d_eh.check_guards("event_hash",&err);
        ok = ok && d_hh.check_guards("heap_hash",&err);
        ok = ok && d_rh.check_guards("root_hash",&err);
        ok = ok && d_eph.check_guards("eph_hash",&err);
        ok = ok && d_rem.check_guards("rem_hash",&err);
        ok = ok && d_ch.check_guards("ctrl_hash",&err);
        ok = ok && d_inv.check_guards("inv",&err);

        std::vector<int32_t> hc = d_counts.download();
        std::vector<uint64_t> heh=d_eh.download(), hhh=d_hh.download(), hrh=d_rh.download();
        std::vector<uint64_t> heph=d_eph.download(), hrem=d_rem.download(), hch=d_ch.download();
        std::vector<int32_t> hinv=d_inv.download();

        IgcwExpected exp;
        igcw_oracle_step(&oracle, op, &exp);

        IgcwGotView got;
        got.counts = hc.data();
        got.gc_event_hash = heh[0];
        got.heap_hash = hhh[0];
        got.root_hash = hrh[0];
        got.ephemeron_hash = heph[0];
        got.remembered_hash = hrem[0];
        got.gc_controller_hash = hch[0];
        got.invalid_flag = hinv[0];

        ok = ok && igcw_check(exp, got, &err);

        OpSnapshot snap; std::memset(&snap,0,sizeof(snap));
        for (int c=0;c<IGCW_NUM_COUNTS;++c) snap.counts[c]=hc[c];
        snap.eh=heh[0]; snap.hh=hhh[0]; snap.rh=hrh[0]; snap.eph=heph[0]; snap.rem=hrem[0]; snap.ch=hch[0]; snap.inv=hinv[0];

        ++(*total);
        if (ok) ++(*passed);
        else {
            all_ok=false;
            if (first_err && first_err->empty()) {
                std::ostringstream o; o<<sc.name<<" op "<<i<<" (opcode "<<op.opcode<<"): "<<err;
                *first_err=o.str();
            }
        }
        if (snaps) snaps->push_back(snap);

        if (verbose && (!ok)) {
            std::printf("  scenario %-30s op %03zu opcode=%d FAIL %s\n", sc.name.c_str(), i, op.opcode, err.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    cudaFree(d_workspace);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_snaps(const std::vector<OpSnapshot>& a, const std::vector<OpSnapshot>& b, std::string* err) {
    if (a.size()!=b.size()) { if(err)*err="op count mismatch"; return false; }
    for (size_t i=0;i<a.size();++i) {
        if (std::memcmp(a[i].counts,b[i].counts,sizeof(a[i].counts))!=0 ||
            a[i].eh!=b[i].eh || a[i].hh!=b[i].hh || a[i].rh!=b[i].rh ||
            a[i].eph!=b[i].eph || a[i].rem!=b[i].rem || a[i].ch!=b[i].ch || a[i].inv!=b[i].inv) {
            if(err){ std::ostringstream o; o<<"replay mismatch at op "<<i; *err=o.str(); }
            return false;
        }
    }
    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        std::vector<Scenario> scenarios = build_scenarios();
        int passed=0, total=0;
        bool all_ok=true;

        for (const Scenario& sc : scenarios) {
            std::vector<OpSnapshot> base, replay;
            std::string err;
            bool ok_base = run_once(sc,true,&base,&passed,&total,&err);
            bool ok_replay = run_once(sc,false,&replay,&passed,&total,&err);

            if (ok_base && ok_replay) {
                std::string ce;
                if (compare_snaps(base,replay,&ce)) {
                    std::printf("scenario %-30s ops=%-3zu exact-replay PASS\n", sc.name.c_str(), sc.ops.size());
                } else {
                    all_ok=false;
                    std::printf("scenario %-30s exact-replay FAIL  %s\n", sc.name.c_str(), ce.c_str());
                }
            } else {
                all_ok=false;
                std::printf("scenario %-30s FAIL  %s\n", sc.name.c_str(), err.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return (all_ok && passed==total) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr,"fatal: %s\n", ex.what());
        return 1;
    }
}
