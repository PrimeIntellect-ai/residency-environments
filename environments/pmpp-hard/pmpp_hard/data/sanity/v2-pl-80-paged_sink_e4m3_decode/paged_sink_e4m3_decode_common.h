// ============================================================================
// file: paged_sink_e4m3_decode_common.h
// ============================================================================

#ifndef PAGED_SINK_E4M3_DECODE_COMMON_H_
#define PAGED_SINK_E4M3_DECODE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define PSE_ABI_VERSION 1

#define PSE_MIN_B 1
#define PSE_MAX_B 64
#define PSE_MIN_HQ 1
#define PSE_MAX_HQ 32
#define PSE_MIN_HKV 1
#define PSE_MAX_HKV 16
#define PSE_MIN_D 64
#define PSE_MAX_D 128
#define PSE_MIN_SINK 0
#define PSE_MAX_SINK 64
#define PSE_MIN_WINDOW 16
#define PSE_MAX_WINDOW 1024
#define PSE_MIN_SEQ_CAP 64
#define PSE_MAX_SEQ_CAP 8192
#define PSE_MIN_PAGES 8
#define PSE_MAX_PAGES 4096
#define PSE_MAX_NEW_TOKENS 8

#define PSE_Y_ATOL 2.0e-3f
#define PSE_Y_RTOL 2.0e-3f
#define PSE_LSE_ATOL 2.0e-3f
#define PSE_LSE_RTOL 2.0e-3f

/*
CONTRACT: paged_sink_e4m3_decode

Stateful single-GPU paged FP8(E4M3) KV-cache decode pipeline with attention
sinks and a sliding window (StreamingLLM-style), grouped-query attention, and
deterministic page lifecycle management.

solution_init allocates all persistent state. solution_run advances exactly
one step (a step may append several tokens per sequence: chunked prefill and
single-token decode use the same entry point). solution_reset returns all
persistent state to the initial configuration.

Legal configuration (ProblemSpec):
  B:            number of logical sequences.        1 <= B <= 64.
  Hq:           query heads.                        1 <= Hq <= 32.
  Hkv:          KV heads.                           1 <= Hkv <= 16, Hq % Hkv == 0.
  GQA group:    group = Hq / Hkv. Query head hq reads KV head hq / group.
  D:            head dimension.                     D in {64, 128}.
  page_size P:  tokens per physical page.           P in {8, 16, 32}.
  n_sink S:     attention-sink prefix length.       0 <= S <= 64.
  window W:     sliding-window length.              16 <= W <= 1024.
  max_seq_len:  hard cap on logical length.         64 <= max_seq_len <= 8192.
  max_pages:    physical page pool size.            8 <= max_pages <= 4096.

Derived:
  max_logical_pages = ceil(max_seq_len / P).

Persistent state (logical model; representation is free):
  - A physical page pool of max_pages pages. Each page stores, for every KV
    head, P token slots of quantized K and V (see storage format).
  - page_table[B, max_logical_pages] int32: physical page id or -1.
  - seq_len[B] int32: tokens appended so far (never exceeds max_seq_len;
    the harness guarantees inputs never overflow it).
  - step_counter int32: starts at 0, incremented at the START of every
    solution_run (even when active_count == 0).
  - total_allocs, total_frees int32: cumulative counters since reset.

Storage format (NORMATIVE, byte-exact):
  Every appended token stores, per KV head h, for K and V separately:
    - one int8 scale exponent  k_scale_exp / v_scale_exp,
    - D quantized FP8-E4M3 bytes.

  E4M3 byte layout: S EEEE MMM, exponent bias 7 (OCP FP8 E4M3):
    E > 0:            value = (-1)^S * 2^(E-7) * (1 + M/8)
    E == 0:           value = (-1)^S * 2^-9 * M        (subnormals; M=0 is +-0)
    E == 15, M == 7:  NaN slot (NEVER produced or stored by this task)
    max finite:       0x7E = +448, 0xFE = -448

  Quantization of one fp32 vector x[0..D) (per token, per KV head, K and V
  independently):
    amax = max_d |x[d]|                      (exact fp32 comparisons)
    if amax == 0:
      scale_exp = 0; every byte = 0x00 if x[d] is +0.0f, 0x80 if -0.0f.
    else:
      kfloor    = floor(log2(amax))          (for normal fp32 this is the
                                              unbiased exponent field; inputs
                                              guarantee amax is normal)
      scale_exp = clamp(kfloor - 8, -110, 110)   stored as int8
      z[d]      = x[d] * 2^(-scale_exp)      (exact: power-of-two scaling;
                                              amax*2^-scale_exp lies in [256,512))
      byte[d]   = e4m3_encode(z[d])

  e4m3_encode(z) for finite fp32 z (NORMATIVE):
    - z == +-0.0f: 0x00 / 0x80 (sign of the zero is preserved).
    - |z| > 448.0f: saturate to sign * 448 (0x7E / 0xFE).
    - otherwise round |z| to the NEAREST representable finite non-negative
      E4M3 value ({0} U subnormals U normals up to 448). On an exact tie
      between two adjacent representable values, choose the one whose
      mantissa field M is even (round-half-to-even on the destination
      significand). The sign bit is always preserved, including when the
      magnitude rounds to zero.
    Inputs guarantee z is never NaN/Inf.

  Dequantization (NORMATIVE):
    dequant(byte, scale_exp) = e4m3_decode(byte) * 2^scale_exp,
    where e4m3_decode is the exact value from the byte layout above.

Input guarantees (harness invariants; implementations may rely on them):
  - new_k / new_v entries are finite fp32, |x| <= 32768, and either 0.0f or
    |x| >= 2^-96. q entries are finite fp32 with |x| <= 16.
  - active_seq entries are unique within a step and in [0, B).
  - new_token_count entries are in [0, PSE_MAX_NEW_TOKENS].
  - seq_len never exceeds max_seq_len.
  - max_pages >= B * (ceil(S/P) + ceil(W/P) + 3): a page allocation never
    finds the pool empty in a conforming implementation.

Per-step RunSpec:
  active_count: rows in this step, 0 <= active_count <= B.
  step_id:      opaque, for debugging only.

Inputs (device pointers):
  active_seq[active_count]                      int32
  new_token_count[active_count]                 int32, in [0, 8]
  new_k[active_count, 8, Hkv, D]                fp32 (unused token slots ignored)
  new_v[active_count, 8, Hkv, D]                fp32
  q[active_count, Hq, D]                        fp32 (this step's queries)

Step semantics (NORMATIVE order):
  (1) step_counter += 1.
  (2) Appends: rows in index order a = 0..active_count-1; within a row,
      tokens in index order nt = 0..new_token_count[a]-1. For each token:
        pos = seq_len[seq]   (before increment)
        lp  = pos / P, off = pos % P
        if page_table[seq][lp] < 0:
            allocate the FREE physical page with the LOWEST id
            (a page is free iff it is not referenced by any page_table entry);
            page_table[seq][lp] = that id; total_allocs += 1.
        quantize new_k[a][nt] / new_v[a][nt] (all Hkv heads) into slot
        (physical page, off); seq_len[seq] += 1.
  (3) Dead-page reclamation: for seq b = 0..B-1 ascending, for lp ascending:
      a RESIDENT page is DEAD iff its token range [lp*P, lp*P + P) satisfies
        lp*P >= sink_end(b)  AND  lp*P + P <= win_start(b)
      (definitions below, evaluated with the post-append seq_len).
      Each dead resident page is freed: page_table[b][lp] = -1,
      total_frees += 1. Pages freed in step t are allocatable from step t+1
      on (all of this step's allocations already happened in phase 2).
  (4) Outputs (all computed from the post-append, post-reclamation state).

Live set (NORMATIVE): for sequence b with L = seq_len[b]:
  sink_end(b)  = min(S, L)
  win_start(b) = max(sink_end(b), L - W)
  live positions = [0, sink_end(b)) U [win_start(b), L), ascending order.
  Every live position is always resident (sink pages never die; window pages
  are never dead), so attention needs no missing-page handling.

Attention: for every active row a (in this step) and query head hq:
    kvh      = hq / (Hq / Hkv)
    score(t) = dot(q[a,hq,:], dequant_K(t, kvh, :)) * 2^k_scale_exp(t,kvh)
               * rsqrt(D)        for every live position t of sequence
               active_seq[a]     (the 2^k_scale_exp factor is written out
               explicitly here; mathematically it is part of dequant)
    m        = max_t score(t)
    l        = sum_t exp(score(t) - m)
    y[a,hq,d]  = sum_t (exp(score(t) - m) / l) * dequant_V(t, kvh, d)
    lse[a,hq]  = m + log(l)      (natural log)
  If the live set is empty (L == 0): y[a,hq,:] = 0 and lse[a,hq] = 0.
  Any numerically stable evaluation order is allowed. The grader compares
    |got - expected| <= ATOL + RTOL * |expected|
  with ATOL = RTOL = 2e-3 for both y and lse.

FNV-1a-64 PRIMITIVE (NORMATIVE -- this project uses a NON-CANONICAL basis;
  the agent MUST use this exact basis or every checksum fails):
    offset basis = 1469598103934665603  (hex 0x14650FB0739D0383) -- this is NOT
      the canonical FNV-1a-64 basis 14695981039346656037 (0xCBF29CE484222325).
    prime        = 1099511628211  (0x100000001B3).
    fold: start h = offset basis; absorb each field's raw bytes little-endian
      at its full declared width (i8=1, i32=4, u64=8, two's complement); per
      byte b: h = (h XOR b) * prime (mod 2^64). No length prefix / terminator
      / final mix.

Outputs (device pointers):
  y[active_count, Hq, D]      fp32   (tolerance above)
  lse[active_count, Hq]       fp32   (tolerance above)
  seq_len[B]                  int32  exact
  kv_hash[B]                  uint64 exact; per sequence b:
      h = basis
      fold seq_len[b] (int32), sink_end(b) (int32), win_start(b) (int32)
      for each live position t of b, ascending:
        for h_kv = 0..Hkv-1:
          fold k_scale_exp(t, h_kv)   (1 int8 byte)
          fold the D K bytes of (t, h_kv)  (D bytes, d ascending)
          fold v_scale_exp(t, h_kv)   (1 int8 byte)
          fold the D V bytes of (t, h_kv)  (D bytes, d ascending)
  page_state_checksum[1]      uint64 exact; global:
      h = basis
      fold as int32 each: B, Hq, Hkv, D, page_size, n_sink, window,
                          max_pages, max_seq_len, step_counter
      for b = 0..B-1 ascending:
        fold seq_len[b] (int32)
        n_lp = ceil(seq_len[b] / P); fold n_lp (int32)
        for lp = 0..n_lp-1: fold page_table[b][lp] (int32, -1 if absent)
      fold total_allocs, total_frees, free_pages (each int32)
  free_pages[1]               int32  exact: pages referenced by no table entry
  total_allocs[1]             int32  exact (cumulative since reset)
  total_frees[1]              int32  exact (cumulative since reset)

Determinism:
  - Exact replay of the same step sequence must reproduce every output bit
    (checksums included) exactly.
  - Permuting the row order within a step must reproduce y / lse / seq_len /
    kv_hash / free_pages / total_allocs / total_frees exactly
    (page_state_checksum may differ: allocation interleaving is row-ordered).

Rules for solution_run:
  - May launch kernels and use cudaMemsetAsync / device-to-device async copies.
  - May not call cudaMalloc/cudaFree (allocate persistent state in
    solution_init; persistent allocations up to ~1 GiB are acceptable).
  - May not perform synchronous host/device copies or otherwise synchronize
    the stream inside solution_run.
  - May not use CUB, Thrust, cuBLAS, cuDNN, or CUTLASS.
  - Must use only the provided workspace plus persistent state.
*/

struct alignas(8) PseProblemSpec {
    int32_t abi_version;
    int32_t B;
    int32_t Hq;
    int32_t Hkv;
    int32_t D;
    int32_t page_size;
    int32_t n_sink;
    int32_t window;
    int32_t max_seq_len;
    int32_t max_pages;
    int32_t flags;
    int32_t reserved[5];
};

struct alignas(8) PseRunSpec {
    int32_t abi_version;
    int32_t active_count;
    int32_t step_id;
    int32_t reserved[13];
};

struct alignas(8) PseInputs {
    const int32_t* active_seq;
    const int32_t* new_token_count;
    const float* new_k;
    const float* new_v;
    const float* q;
};

struct alignas(8) PseOutputs {
    float* y;
    float* lse;
    int32_t* seq_len;
    uint64_t* kv_hash;
    uint64_t* page_state_checksum;
    int32_t* free_pages;
    int32_t* total_allocs;
    int32_t* total_frees;
};

static_assert(sizeof(PseProblemSpec) == 64, "PseProblemSpec layout drift");
static_assert(sizeof(PseRunSpec) == 64, "PseRunSpec layout drift");
static_assert(sizeof(PseInputs) == 40, "PseInputs layout drift");
static_assert(sizeof(PseOutputs) == 64, "PseOutputs layout drift");

static_assert(offsetof(PseProblemSpec, abi_version) == 0, "layout");
static_assert(offsetof(PseProblemSpec, B) == 4, "layout");
static_assert(offsetof(PseProblemSpec, Hq) == 8, "layout");
static_assert(offsetof(PseProblemSpec, Hkv) == 12, "layout");
static_assert(offsetof(PseProblemSpec, D) == 16, "layout");
static_assert(offsetof(PseProblemSpec, page_size) == 20, "layout");
static_assert(offsetof(PseProblemSpec, n_sink) == 24, "layout");
static_assert(offsetof(PseProblemSpec, window) == 28, "layout");
static_assert(offsetof(PseProblemSpec, max_seq_len) == 32, "layout");
static_assert(offsetof(PseProblemSpec, max_pages) == 36, "layout");

static_assert(offsetof(PseRunSpec, abi_version) == 0, "layout");
static_assert(offsetof(PseRunSpec, active_count) == 4, "layout");
static_assert(offsetof(PseRunSpec, step_id) == 8, "layout");

static inline size_t pse_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int pse_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int pse_valid_D(int D) {
    return D == 64 || D == 128;
}

static inline int pse_valid_page_size(int P) {
    return P == 8 || P == 16 || P == 32;
}

static inline int pse_max_logical_pages(int max_seq_len, int P) {
    return (max_seq_len + P - 1) / P;
}

static inline int pse_validate_problem_spec(const PseProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != PSE_ABI_VERSION) return 0;
    if (spec->B < PSE_MIN_B || spec->B > PSE_MAX_B) return 0;
    if (spec->Hq < PSE_MIN_HQ || spec->Hq > PSE_MAX_HQ) return 0;
    if (spec->Hkv < PSE_MIN_HKV || spec->Hkv > PSE_MAX_HKV) return 0;
    if (spec->Hq % spec->Hkv != 0) return 0;
    if (!pse_valid_D(spec->D)) return 0;
    if (!pse_valid_page_size(spec->page_size)) return 0;
    if (spec->n_sink < PSE_MIN_SINK || spec->n_sink > PSE_MAX_SINK) return 0;
    if (spec->window < PSE_MIN_WINDOW || spec->window > PSE_MAX_WINDOW) return 0;
    if (spec->max_seq_len < PSE_MIN_SEQ_CAP || spec->max_seq_len > PSE_MAX_SEQ_CAP) return 0;
    if (spec->max_pages < PSE_MIN_PAGES || spec->max_pages > PSE_MAX_PAGES) return 0;
    return 1;
}

static inline int pse_validate_run_spec(const PseRunSpec* run, const PseProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != PSE_ABI_VERSION) return 0;
    if (run->active_count < 0 || run->active_count > spec->B) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const PseProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const PseProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const PseRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // PAGED_SINK_E4M3_DECODE_COMMON_H_
