// ============================================================================
// file: paged_cow_prefill_decode_common.h
// ============================================================================

#ifndef PAGED_COW_PREFILL_DECODE_COMMON_H_
#define PAGED_COW_PREFILL_DECODE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define CPD_ABI_VERSION 1

#define CPD_MIN_B 1
#define CPD_MAX_B 32
#define CPD_MIN_HQ 1
#define CPD_MAX_HQ 32
#define CPD_MIN_HKV 1
#define CPD_MAX_HKV 8
#define CPD_MIN_SEQ_CAP 64
#define CPD_MAX_SEQ_CAP 8192
#define CPD_MIN_PAGES 8
#define CPD_MAX_PAGES 4096

#define CPD_OP_APPEND 0
#define CPD_OP_FORK_APPEND 1
#define CPD_OP_RELEASE 2

#define CPD_Y_ATOL 2.5e-3f
#define CPD_Y_RTOL 2.5e-3f
#define CPD_LSE_ATOL 2.5e-3f
#define CPD_LSE_RTOL 2.5e-3f

/*
CONTRACT: paged_cow_prefill_decode

Stateful single-GPU continuous-batching attention engine (vLLM-style) over a
REFCOUNTED shared paged fp16 KV cache with copy-on-write forks. One
solution_run = one scheduler step over a mixed batch: some rows run chunked
PREFILL (up to max_chunk tokens appended, one causal attention output per
appended token), some run single-token DECODE (the same append path with one
token), some FORK a fresh sequence id from a live sequence (sharing all its
full pages by refcount, eagerly copying only the partial tail page), and
some RELEASE a sequence (dropping its page references; pages are freed when
their reference count reaches zero and become reusable in the SAME step).

solution_init allocates all persistent state. solution_reset returns all
persistent state to the initial configuration.

Legal configuration (CpdProblemSpec):
  B:            number of sequence ids.       1 <= B <= 32.
  Hq:           query heads.                  1 <= Hq <= 32.
  Hkv:          KV heads.                     1 <= Hkv <= 8, Hq % Hkv == 0.
  GQA group:    group = Hq / Hkv. Query head hq reads KV head hq / group.
  D:            head dimension.               D in {64, 128}.
  page_size P:  tokens per physical page.     P in {8, 16, 32}.
  max_chunk C:  prefill chunk capacity.       C in {16, 32, 64}.
  max_seq_len:  hard cap on logical length.   64 <= max_seq_len <= 8192.
  max_pages:    physical page pool size.      8 <= max_pages <= 4096.

Persistent state (logical model; representation is free):
  - A physical page pool of max_pages pages. Each page stores, for every KV
    head, P token slots of fp16 K and V (storage format below).
  - page_table[B, ceil(max_seq_len/P)] int32: physical page id or -1.
    DIFFERENT sequences MAY reference the SAME physical page (sharing).
  - refcount(p) = number of page_table entries referencing page p. A page
    is FREE iff refcount(p) == 0.
  - seq_len[B] int32.
  - step_counter, total_allocs, total_frees, total_forks, total_releases
    int32 (cumulative since reset).

  INVARIANT (maintained by conforming implementations): a page referenced by
  more than one sequence is always FULL (all P slots written) and is never
  written again; every partial tail page has refcount 1.

Storage format (NORMATIVE, byte-exact):
  Every appended token stores, per KV head h, D fp16 K values and D fp16 V
  values. Conversion from the fp32 inputs is IEEE 754 binary16
  round-to-nearest-even (identical to CUDA __float2half_rn), including
  subnormal halves and signed zeros. Inputs guarantee |x| <= 1024, so
  overflow to infinity never occurs.

FNV-1a-64 PRIMITIVE (NORMATIVE -- this project uses a NON-CANONICAL basis;
  the agent MUST use this exact basis or every checksum fails):
    offset basis = 1469598103934665603  (hex 0x14650FB0739D0383) -- this is NOT
      the canonical FNV-1a-64 basis 14695981039346656037 (0xCBF29CE484222325).
    prime        = 1099511628211  (0x100000001B3).
    fold: start h = offset basis; absorb each field's raw bytes little-endian
      at its full declared width (i8=1, u16=2, i32=4, u64=8, two's
      complement); per byte b: h = (h XOR b) * prime (mod 2^64). No length
      prefix / terminator / final mix.

TOKEN FOLD (NORMATIVE, used by kv_hash):
  for h_kv = 0..Hkv-1:
    for d = 0..D-1: fold the fp16 K bit pattern of (h_kv, d) as u16 (2 LE bytes)
    for d = 0..D-1: fold the fp16 V bit pattern of (h_kv, d) as u16

Input guarantees (harness invariants; implementations may rely on them):
  - new_k/new_v entries of real token slots are finite fp32 with |x| <= 1024
    and either 0.0f or |x| >= 2^-90. q entries are finite with |x| <= 4.
    Padding slots (nt >= new_token_count[a]) contain arbitrary finite values
    and must not affect any output.
  - active_seq entries are unique within a step and in [0, B).
  - op_kind[a] in {0 (APPEND), 1 (FORK_APPEND), 2 (RELEASE)}.
  - APPEND rows: new_token_count in [1, C]; the sequence may be empty (a
    fresh prefill) or live.
  - FORK_APPEND rows: fork_src[a] in [0, B) is a live sequence
    (seq_len >= 1) that is NOT released or forked-into in this step and is
    not the row's own sequence; the row's own sequence is EMPTY (seq_len 0,
    no page references); new_token_count in [0, C].
    seq_len[src] + appended tokens never exceeds max_seq_len.
  - RELEASE rows: the sequence is live (seq_len >= 1); new_token_count == 0.
  - fork_src is ignored for non-fork rows.
  - seq_len never exceeds max_seq_len.
  - The pool is sized so that a conforming implementation never finds it
    empty at any allocation (including copy-on-write tail copies).

Per-step RunSpec:
  active_count: rows in this step, 0 <= active_count <= B.
  step_id:      opaque, for debugging only.

Inputs (device pointers, C = max_chunk):
  active_seq[active_count]                 int32
  op_kind[active_count]                    int32
  fork_src[active_count]                   int32
  new_token_count[active_count]            int32
  new_k[active_count, C, Hkv, D]           fp32 (pad slots ignored)
  new_v[active_count, C, Hkv, D]           fp32
  q[active_count, C, Hq, D]                fp32 (per appended token queries)

Step semantics (NORMATIVE order):
  (1) step_counter += 1.
  (2) RELEASES, rows in index order: for each RELEASE row with sequence b:
      for lp = 0..ceil(seq_len[b]/P)-1 ascending: drop the reference
      (page_table[b][lp] = -1); if the page's refcount is now 0 it is freed
      (total_frees += 1) and is allocatable later in this SAME step.
      seq_len[b] = 0; the sequence's running-hash state returns to the
      initial value; total_releases += 1.
  (3) FORKS, rows in index order: for each FORK_APPEND row with sequence b
      and source s (state as left by phase 2):
        seq_len[b] = seq_len[s]; total_forks += 1.
        For every FULL logical page of s (lp*P + P <= seq_len[s]):
          page_table[b][lp] = page_table[s][lp]  (shared; refcount grows).
        If seq_len[s] % P != 0 (partial tail): allocate the FREE page with
        the LOWEST id (total_allocs += 1), COPY the tail page's stored bytes
        for the occupied slots (positions lp*P..seq_len[s]-1), and reference
        the COPY: page_table[b][tail_lp] = new page. The source keeps its
        original tail. (Eager copy-on-write: the fork may append privately.)
      Forks observe the source PRE-APPEND (phase 4 has not run).
  (4) APPENDS, rows in index order (APPEND and FORK_APPEND rows with
      new_token_count >= 1): tokens nt = 0..new_token_count[a]-1 appended at
      pos = seq_len_pre_append[b] + nt: if page_table[b][pos/P] < 0,
      allocate the FREE page with the LOWEST id (total_allocs += 1).
      Convert new_k[a][nt] / new_v[a][nt] (all Hkv heads) to fp16 and store
      at slot pos%P. seq_len[b] += new_token_count[a]. Appends never write a
      shared page (see invariant; the harness never constructs a violating
      schedule for a conforming implementation).
  (5) Attention (post-append state): for every APPEND / FORK_APPEND row a,
      real token slot nt at position p = seq_len_pre_append + nt, and query
      head hq:
        kvh = hq / (Hq / Hkv)
        score(t) = dot(q[a,nt,hq,:], fp32(K[t, kvh, :])) * rsqrt(D)
                   for t = 0..p (the token attends to the whole logical
                   prefix INCLUDING itself; the set is never empty)
        m = max score, l = sum exp(score - m)
        y[a,nt,hq,d] = sum (exp(score-m)/l) * fp32(V[t, kvh, d])
        lse[a,nt,hq] = m + log(l)
      fp32(.) is the exact fp16 -> fp32 conversion. Any numerically stable
      evaluation order is allowed. Grader tolerance:
      |got - expected| <= ATOL + RTOL * |expected|, ATOL = RTOL = 2.5e-3.
  (6) Pad slots (nt >= new_token_count[a], and ALL slots of RELEASE rows):
      y and lse must be written as exact 0.0f.
  (7) Outputs reflect the post-step state and are written on EVERY step,
      including active_count == 0.

Outputs (device pointers):
  y[active_count, C, Hq, D]     fp32   (tolerance above; pad slots exact 0)
  lse[active_count, C, Hq]      fp32   (tolerance above; pad slots exact 0)
  seq_len[B]                    int32  exact
  kv_hash[B]                    uint64 exact; per sequence b:
      h = basis
      for each logical token t = 0..seq_len[b]-1 ascending: TOKEN FOLD
      fold seq_len[b] (int32)
      (seq_len folds LAST: the running hash over the append-only logical
      stream is maintainable incrementally, is INHERITED VERBATIM by forks
      -- the shared prefix and the copied tail are byte-identical -- and
      resets on release)
  page_state_checksum[1]        uint64 exact; global:
      h = basis
      fold as int32 each: B, Hq, Hkv, D, page_size, max_chunk, max_seq_len,
                          max_pages, step_counter
      for b = 0..B-1 ascending:
        fold seq_len[b] (int32)
        n_lp = ceil(seq_len[b] / P); fold n_lp (int32)
        for lp = 0..n_lp-1: fold page_table[b][lp] (int32)
      fold total_allocs, total_frees, total_forks, total_releases,
           free_pages (each int32)
  free_pages[1]                 int32  exact (pages with refcount 0)
  total_allocs[1]               int32  exact (cumulative since reset)
  total_frees[1]                int32  exact
  total_forks[1]                int32  exact
  total_releases[1]             int32  exact

Determinism:
  - Exact replay of the same step sequence must reproduce every output bit
    (checksums included) exactly.
  - Permuting the row order within a step must reproduce y / lse / seq_len /
    kv_hash / free_pages and all cumulative counters exactly
    (page_state_checksum may differ: allocation interleaving is row-ordered
    by contract; the harness never schedules two rows in one step whose
    RESULTING LOGICAL STATE depends on row order).

Rules for solution_run:
  - May launch kernels and use cudaMemsetAsync / device-to-device async copies.
  - May not call cudaMalloc/cudaFree (allocate persistent state in
    solution_init; persistent allocations up to ~1 GiB are acceptable).
  - May not perform synchronous host/device copies or otherwise synchronize
    the stream inside solution_run.
  - May not use CUB, Thrust, cuBLAS, cuDNN, or CUTLASS.
  - Must use only the provided workspace plus persistent state.
*/

struct alignas(8) CpdProblemSpec {
    int32_t abi_version;
    int32_t B;
    int32_t Hq;
    int32_t Hkv;
    int32_t D;
    int32_t page_size;
    int32_t max_chunk;
    int32_t max_seq_len;
    int32_t max_pages;
    int32_t flags;
    int32_t reserved[6];
};

struct alignas(8) CpdRunSpec {
    int32_t abi_version;
    int32_t active_count;
    int32_t step_id;
    int32_t reserved[13];
};

struct alignas(8) CpdInputs {
    const int32_t* active_seq;
    const int32_t* op_kind;
    const int32_t* fork_src;
    const int32_t* new_token_count;
    const float* new_k;
    const float* new_v;
    const float* q;
};

struct alignas(8) CpdOutputs {
    float* y;
    float* lse;
    int32_t* seq_len;
    uint64_t* kv_hash;
    uint64_t* page_state_checksum;
    int32_t* free_pages;
    int32_t* total_allocs;
    int32_t* total_frees;
    int32_t* total_forks;
    int32_t* total_releases;
};

static_assert(sizeof(CpdProblemSpec) == 64, "CpdProblemSpec layout drift");
static_assert(sizeof(CpdRunSpec) == 64, "CpdRunSpec layout drift");
static_assert(sizeof(CpdInputs) == 56, "CpdInputs layout drift");
static_assert(sizeof(CpdOutputs) == 80, "CpdOutputs layout drift");

static_assert(offsetof(CpdProblemSpec, abi_version) == 0, "layout");
static_assert(offsetof(CpdProblemSpec, B) == 4, "layout");
static_assert(offsetof(CpdProblemSpec, Hq) == 8, "layout");
static_assert(offsetof(CpdProblemSpec, Hkv) == 12, "layout");
static_assert(offsetof(CpdProblemSpec, D) == 16, "layout");
static_assert(offsetof(CpdProblemSpec, page_size) == 20, "layout");
static_assert(offsetof(CpdProblemSpec, max_chunk) == 24, "layout");
static_assert(offsetof(CpdProblemSpec, max_seq_len) == 28, "layout");
static_assert(offsetof(CpdProblemSpec, max_pages) == 32, "layout");

static_assert(offsetof(CpdRunSpec, abi_version) == 0, "layout");
static_assert(offsetof(CpdRunSpec, active_count) == 4, "layout");
static_assert(offsetof(CpdRunSpec, step_id) == 8, "layout");

static inline size_t cpd_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int cpd_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int cpd_valid_D(int D) {
    return D == 64 || D == 128;
}

static inline int cpd_valid_page_size(int P) {
    return P == 8 || P == 16 || P == 32;
}

static inline int cpd_valid_max_chunk(int C) {
    return C == 16 || C == 32 || C == 64;
}

static inline int cpd_max_logical_pages(int max_seq_len, int P) {
    return (max_seq_len + P - 1) / P;
}

static inline int cpd_validate_problem_spec(const CpdProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != CPD_ABI_VERSION) return 0;
    if (spec->B < CPD_MIN_B || spec->B > CPD_MAX_B) return 0;
    if (spec->Hq < CPD_MIN_HQ || spec->Hq > CPD_MAX_HQ) return 0;
    if (spec->Hkv < CPD_MIN_HKV || spec->Hkv > CPD_MAX_HKV) return 0;
    if (spec->Hq % spec->Hkv != 0) return 0;
    if (!cpd_valid_D(spec->D)) return 0;
    if (!cpd_valid_page_size(spec->page_size)) return 0;
    if (!cpd_valid_max_chunk(spec->max_chunk)) return 0;
    if (spec->max_seq_len < CPD_MIN_SEQ_CAP || spec->max_seq_len > CPD_MAX_SEQ_CAP) return 0;
    if (spec->max_pages < CPD_MIN_PAGES || spec->max_pages > CPD_MAX_PAGES) return 0;
    return 1;
}

static inline int cpd_validate_run_spec(const CpdRunSpec* run, const CpdProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != CPD_ABI_VERSION) return 0;
    if (run->active_count < 0 || run->active_count > spec->B) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const CpdProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const CpdProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const CpdRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // PAGED_COW_PREFILL_DECODE_COMMON_H_
