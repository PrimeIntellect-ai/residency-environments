// ============================================================================
// file: mk_priority_preempt_common.h
// ============================================================================

#ifndef MK_PRIORITY_PREEMPT_COMMON_H_
#define MK_PRIORITY_PREEMPT_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MKP_ABI_VERSION 1

#define MKP_MIN_M 16384
#define MKP_MAX_M 32768
#define MKP_MIN_R 8
#define MKP_MAX_R 32
#define MKP_MIN_NJOBS 64
#define MKP_MAX_NJOBS 1024
#define MKP_MAX_LEN 64
#define MKP_MAX_JOBS_TOTAL 4096
#define MKP_ROWS 256  // rows per slice (fixed)

// Salt mixing constants (normative).
#define MKP_SALT_ROUND 0x9E3779B9u
#define MKP_SALT_TRACE 0x85EBCA6Bu
#define MKP_SALT_JOB 0xC2B2AE35u
#define MKP_SALT_SLICE 0x27D4EB2Fu

// FNV-1a-64 (NON-canonical basis used across this project).
#define MKP_FNV_BASIS 1469598103934665603ULL
#define MKP_FNV_PRIME 1099511628211ULL

enum MkpDistributionId : int32_t {
    MKP_DIST_UNIFORM = 0,
    MKP_DIST_STAGGER = 1,
    MKP_DIST_LONGTAIL = 2,
    MKP_DIST_BURST = 3,
    MKP_DIST_ZERO_X = 4
};

/*
CONTRACT: mk_priority_preempt

A stateful multi-queue priority scheduler with quantum preemption (mk_*
megakernel-systems family). Persistent state: a job table (up to
MKP_MAX_JOBS_TOTAL jobs per reset epoch) holding, for every live job, its
class, its remaining slice count, its arrival round, and its FULL int8
weight row w (stored at ingestion; jobs routinely outlive the run that
ingested them and keep executing against later runs' X data). Each
solution_run ingests njobs new jobs and executes R scheduling rounds.
All arithmetic is exact integer; there is no tolerance anywhere.

Legal shape family (RunSpec):
  M      in [16384, 32768], multiple of 256   (X rows; may change per run)
  K      in {128, 256}                        (feature dim)
  Q      in {2, 3, 4}                         (priority classes; 0 = highest)
  R      in [8, 32]                           (rounds this run; per run)
  quantum in {1, 2, 4, 8}                     (max slices per execution)
  njobs  in [64, 1024]                        (new jobs this run; per run)

State lifecycle:
  solution_init : allocate for the ProblemSpec maxima.
  solution_reset: job table empty, global job counter := 0.
  solution_run  : ingest + R rounds (see below).
  The grader calls solution_reset before changing K or Q; M, R, quantum,
  njobs and all input bytes may change every run. The grader never
  ingests more than MKP_MAX_JOBS_TOTAL jobs between resets, and job
  descriptors are always valid (class in [0,Q), arrival in [0,R) of the
  ingesting run, len in [1, MKP_MAX_LEN]).

Global job ids: the j-th job ever ingested since reset (run order, then
descriptor order) has gjid == j. New jobs this run: gjid = base + j.

Inputs (device pointers, fully initialized by the harness; all may change
every run):
  X        : int8_t [M, K] row-major.
  jarrival : int32_t [njobs], arrival round of each new job.
  jclass   : int32_t [njobs].
  jlen     : int32_t [njobs], slice count (initial remaining).
  jw       : int8_t [njobs, K] row-major, per-job weight rows (must be
             stored persistently; carryover jobs use their stored copy).

RUN SEMANTICS (normative, exact). Let B = M / 256.

Ingestion (before round 0): new jobs enter the table with rem = len.
A carryover job (ingested by an earlier run, rem > 0) has arrival 0 for
this run. A new job arrives in round jarrival[j].

Round r = 0 .. R-1:
  pending(r) = jobs with rem > 0 whose arrival <= r.
  If pending(r) is empty, the round executes nothing (no trace entries).
  Otherwise c(r) = the smallest class with at least one pending job
  ("strict priority": lower classes preempt the machine between rounds).
  The execution list of round r = pending jobs of class c(r), gjid
  ascending. Every listed job executes exactly once this round:
  execution e (0-based list position) has global trace index
  g = (total executions in rounds 0..r-1 of THIS run) + e, and runs
  n_s = min(quantum, rem) slices ("quantum preemption": the job then
  yields; rem -= n_s after the round).

  Slice s in [0, n_s) of execution (g, job J):
    salt = MKP_SALT_ROUND * (uint32_t)r
         + MKP_SALT_TRACE * (uint32_t)g
         + MKP_SALT_JOB   * (uint32_t)gjid(J)
         + MKP_SALT_SLICE * (uint32_t)s          (uint32 wrap)
    sb[m] = byte m of salt, little-endian (m = 0..3)
    b = (g + s) mod B; the slice reads X rows [b*256, (b+1)*256).
    For i in [0, 256), row = b*256 + i:
      res[i] = sum over k in [0, K) of
                 int32( (int8_t)((uint8_t)X[row*K + k] ^ sb[k & 3]) )
               * int32( w_J[k] )                (exact int32)
      y[row] += (uint64_t)(uint32_t)res[i]      (uint64 wraparound; the
                accumulation is commutative, so any execution order gives
                identical bytes)
    subdig[t], t in [0, 16): FNV-1a-64 over res[16t .. 16t+15] (4 bytes LE
      each, i ascending), seeded MKP_FNV_BASIS.
    slice_digest(s) = FNV over subdig[0..16) (8 bytes LE each), seeded
      MKP_FNV_BASIS.
  exec_digest[g] = FNV over slice_digest[0..n_s) (8 bytes LE each),
      seeded MKP_FNV_BASIS.
  trace[g] = (uint64) gjid
           | (uint64)r      << 24
           | (uint64)c(r)   << 32
           | (uint64)n_s    << 36
           | (uint64)(rem - n_s) << 44          (bits 60..63 zero)

  FNV-1a-64 byte step: h = (h XOR byte) * MKP_FNV_PRIME (mod 2^64);
  multi-byte values absorbed little-endian, full width.

End of run: every live job's arrival becomes 0 (carryover), the global
job counter grows by njobs.

Outputs after EVERY run:
  trace_len   : int32 [1], total executions T this run.
  trace       : uint64 [max_R * MKP_MAX_JOBS_TOTAL cap]; [0, T) specified.
  exec_digest : uint64 [same cap]; [0, T) specified.
  y           : uint64 [M]. y is THIS RUN's accumulation only (starts at
                all zeros every run).
  queue_len   : int32 [Q], live jobs (rem > 0) per class after round R-1.
  queue_dump  : uint64 [MKP_MAX_JOBS_TOTAL cap], live jobs in (class
                ascending, gjid ascending) order:
                  word = gjid | (uint64)class << 24 | (uint64)rem << 32.
                Entries beyond the total live count are unspecified.
  state_checksum : uint64 [1], three-level FNV over the post-run state:
      job_digest(J)  = FNV over gjid (4 bytes LE), class (4 bytes LE),
                       rem (4 bytes LE), then w_J bytes [0, K).
      class_digest(c)= FNV over live-count(c) (4 bytes LE) then
                       job_digest of its live jobs, gjid ascending
                       (8 bytes LE each).
      state_checksum = FNV over class_digest[0..Q) (8 bytes LE each).

ABI:
  The grader passes MkpRunSpec*, MkpInputs*, MkpOutputs* through the
  generic pipeline ABI. MkpProblemSpec carries maximum bounds.

Rules for solution_run:
  - May launch kernels and use cudaMemsetAsync.
  - May use CUDA Graphs (stream capture of this run's own kernels, cached
    across runs) and/or persistent cooperative kernels
    (cudaLaunchCooperativeKernel + grid sync). This matters: R rounds of
    small scheduling kernels are launch-bound, and per-launch overhead is
    a first-order cost. Kernel fusion is equally legal.
  - May not call cudaMalloc/cudaFree (persistent allocations belong in
    solution_init).
  - May not perform synchronous host/device copies and may not
    synchronize the stream to inspect device values on the host.
  - May not use CUB, Thrust, cuBLAS, cuDNN, or CUTLASS.
  - Must be deterministic: identical reset+run sequences must produce
    identical bytes in every specified output region.
*/

struct alignas(8) MkpProblemSpec {
    int32_t abi_version;
    int32_t max_M;
    int32_t max_K;
    int32_t max_Q;
    int32_t max_R;
    int32_t max_njobs;
    int32_t flags;
    int32_t reserved[9];
};

struct alignas(8) MkpRunSpec {
    int32_t abi_version;
    int32_t M;
    int32_t K;
    int32_t Q;
    int32_t R;
    int32_t quantum;
    int32_t njobs;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[6];
};

struct alignas(8) MkpInputs {
    const int8_t* X;
    const int32_t* jarrival;
    const int32_t* jclass;
    const int32_t* jlen;
    const int8_t* jw;
    const void* reserved0;
};

struct alignas(8) MkpOutputs {
    int32_t* trace_len;      // [1]
    uint64_t* trace;         // [max_R * MKP_MAX_JOBS_TOTAL]
    uint64_t* exec_digest;   // [max_R * MKP_MAX_JOBS_TOTAL]
    uint64_t* y;             // [M]
    int32_t* queue_len;      // [Q]
    uint64_t* queue_dump;    // [MKP_MAX_JOBS_TOTAL]
    uint64_t* state_checksum;// [1]
    void* reserved0;
};

static_assert(sizeof(MkpProblemSpec) == 64, "MkpProblemSpec layout drift");
static_assert(sizeof(MkpRunSpec) == 64, "MkpRunSpec layout drift");
static_assert(sizeof(MkpInputs) == 48, "MkpInputs layout drift");
static_assert(sizeof(MkpOutputs) == 64, "MkpOutputs layout drift");

static_assert(offsetof(MkpRunSpec, M) == 4, "layout");
static_assert(offsetof(MkpRunSpec, K) == 8, "layout");
static_assert(offsetof(MkpRunSpec, Q) == 12, "layout");
static_assert(offsetof(MkpRunSpec, R) == 16, "layout");
static_assert(offsetof(MkpRunSpec, quantum) == 20, "layout");
static_assert(offsetof(MkpRunSpec, njobs) == 24, "layout");

static inline size_t mkp_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int mkp_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int mkp_valid_K(int K) { return K == 128 || K == 256; }
static inline int mkp_valid_Q(int Q) { return Q >= 2 && Q <= 4; }
static inline int mkp_valid_quantum(int q) {
    return q == 1 || q == 2 || q == 4 || q == 8;
}

static inline int mkp_validate_problem_spec(const MkpProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MKP_ABI_VERSION) return 0;
    if (spec->max_M < MKP_MIN_M || spec->max_M > MKP_MAX_M) return 0;
    if (spec->max_M % 256 != 0) return 0;
    if (!mkp_valid_K(spec->max_K)) return 0;
    if (!mkp_valid_Q(spec->max_Q)) return 0;
    if (spec->max_R < MKP_MIN_R || spec->max_R > MKP_MAX_R) return 0;
    if (spec->max_njobs < MKP_MIN_NJOBS ||
        spec->max_njobs > MKP_MAX_NJOBS) return 0;
    return 1;
}

static inline int mkp_validate_run_spec(const MkpRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != MKP_ABI_VERSION) return 0;
    if (run->M < MKP_MIN_M || run->M > MKP_MAX_M) return 0;
    if (run->M % 256 != 0) return 0;
    if (!mkp_valid_K(run->K)) return 0;
    if (!mkp_valid_Q(run->Q)) return 0;
    if (run->R < MKP_MIN_R || run->R > MKP_MAX_R) return 0;
    if (!mkp_valid_quantum(run->quantum)) return 0;
    if (run->njobs < MKP_MIN_NJOBS || run->njobs > MKP_MAX_NJOBS) return 0;
    if (run->distribution_id < MKP_DIST_UNIFORM ||
        run->distribution_id > MKP_DIST_ZERO_X) return 0;
    return 1;
}

static __host__ __device__ inline uint64_t mkp_trace_word(
    uint32_t gjid, int r, int c, int n_s, uint32_t rem_after) {
    return (uint64_t)gjid |
           ((uint64_t)(uint32_t)r << 24) |
           ((uint64_t)(uint32_t)c << 32) |
           ((uint64_t)(uint32_t)n_s << 36) |
           ((uint64_t)(rem_after & 0xFFFFu) << 44);
}

static __host__ __device__ inline uint64_t mkp_dump_word(
    uint32_t gjid, int c, uint32_t rem) {
    return (uint64_t)gjid |
           ((uint64_t)(uint32_t)c << 24) |
           ((uint64_t)(rem & 0xFFFFu) << 32);
}

extern "C" size_t solution_workspace_bytes(const MkpProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const MkpProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const MkpRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // MK_PRIORITY_PREEMPT_COMMON_H_
