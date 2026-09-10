// ============================================================================
// file: spec_tree_verify_e5m2_kv_common.h
// ============================================================================

#ifndef SPEC_TREE_VERIFY_E5M2_KV_COMMON_H_
#define SPEC_TREE_VERIFY_E5M2_KV_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define STV_ABI_VERSION 1

#define STV_MIN_B 1
#define STV_MAX_B 32
#define STV_MIN_HQ 1
#define STV_MAX_HQ 32
#define STV_MIN_HKV 1
#define STV_MAX_HKV 8
#define STV_MIN_SEQ_CAP 64
#define STV_MAX_SEQ_CAP 8192
#define STV_MIN_PAGES 8
#define STV_MAX_PAGES 4096
#define STV_MAX_NODES_LIMIT 64

#define STV_Y_ATOL 2.5e-3f
#define STV_Y_RTOL 2.5e-3f
#define STV_LSE_ATOL 2.5e-3f
#define STV_LSE_RTOL 2.5e-3f

/*
CONTRACT: spec_tree_verify_e5m2_kv

Stateful single-GPU speculative-decoding TREE VERIFICATION engine over a
paged FP8(E5M2)-quantized committed KV cache (SpecInfer/Medusa-style token
trees, grouped-query attention). One solution_run = one speculation round
per active sequence:

  a draft forest of up to max_nodes K/V/query triples arrives; every node is
  quantized into transient SCRATCH pages drawn from the same physical pool
  as the committed cache; every node's attention output over
  (committed prefix + its ancestor path, including itself) is produced;
  nodes are then verified by comparing the byte-exact FNV signature of their
  quantized bytes against a supplied target signature; the deepest fully
  accepted chain plus one unconditional bonus token is committed to the
  paged cache; all scratch pages are released.

Because acceptance is defined on the QUANTIZED bytes, any deviation in the
normative quantizer changes the control flow (accepted chains and committed
lengths), not just the numerics.

solution_init allocates all persistent state. solution_reset returns all
persistent state to the initial configuration.

Legal configuration (StvProblemSpec):
  B:            number of logical sequences.  1 <= B <= 32.
  Hq:           query heads.                  1 <= Hq <= 32.
  Hkv:          KV heads.                     1 <= Hkv <= 8, Hq % Hkv == 0.
  GQA group:    group = Hq / Hkv. Query head hq reads KV head hq / group.
  D:            head dimension.               D in {64, 128}.
  page_size P:  tokens per physical page.     P in {8, 16, 32}.
  max_nodes N:  draft-tree capacity.          N in {16, 32, 64}.
  max_seq_len:  hard cap on committed length. 64 <= max_seq_len <= 8192.
  max_pages:    physical page pool size.      8 <= max_pages <= 4096.

Persistent state (logical model; representation is free):
  - A physical page pool of max_pages pages holding quantized K/V tokens
    (storage format below). A page is FREE iff it is neither referenced by
    any committed page-table entry nor currently held as scratch.
  - Committed page_table[B, ceil(max_seq_len/P)] int32: page id or -1.
  - seq_len[B] int32 committed tokens.
  - step_counter, total_allocs, total_frees int32 (cumulative since reset;
    scratch and committed allocations both count).

Storage format (NORMATIVE, byte-exact):
  Every stored token (scratch node, committed token, bonus) stores per KV
  head h: one int8 scale exponent k_scale_exp, D FP8-E5M2 K bytes, one int8
  v_scale_exp, D FP8-E5M2 V bytes.

  E5M2 byte layout: S EEEEE MM, exponent bias 15:
    E > 0, E < 31:  value = (-1)^S * 2^(E-15) * (1 + M/4)
    E == 0:         value = (-1)^S * M * 2^-16   (subnormals; M=0 is +-0)
    E == 31:        Inf/NaN slots (NEVER produced or stored by this task)
    max finite:     0x7B = +57344, 0xFB = -57344

  Quantization of one fp32 vector x[0..D) (per token, per KV head, K and V
  independently):
    amax = max_d |x[d]|                      (exact fp32 comparisons)
    if amax == 0:
      scale_exp = 0; every byte = 0x00 if x[d] is +0.0f, 0x80 if -0.0f.
    else:
      kfloor    = floor(log2(amax))          (unbiased exponent field;
                                              inputs guarantee amax normal)
      scale_exp = clamp(kfloor - 15, -110, 110)   stored as int8
      z[d]      = x[d] * 2^(-scale_exp)      (exact power-of-two scaling;
                                              amax*2^-scale_exp in [32768,65536))
      byte[d]   = e5m2_encode(z[d])

  e5m2_encode(z) for finite fp32 z (NORMATIVE):
    - z == +-0.0f: 0x00 / 0x80 (sign of the zero is preserved).
    - |z| > 57344.0f: saturate to sign * 57344 (0x7B / 0xFB). (Reachable:
      the amax element itself saturates whenever its scaled value exceeds
      57344.)
    - otherwise round |z| to the NEAREST representable finite non-negative
      E5M2 value ({0} U subnormals U normals up to 57344). On an exact tie
      between two adjacent representable values, choose the one whose
      mantissa field M is even (round-half-to-even on the destination
      significand). The sign bit is always preserved, including when the
      magnitude rounds to zero.
    Inputs guarantee z is never NaN/Inf.

  Dequantization (NORMATIVE):
    dequant(byte, scale_exp) = e5m2_decode(byte) * 2^scale_exp.

FNV-1a-64 PRIMITIVE (NORMATIVE -- this project uses a NON-CANONICAL basis;
  the agent MUST use this exact basis or every checksum fails):
    offset basis = 1469598103934665603  (hex 0x14650FB0739D0383) -- this is NOT
      the canonical FNV-1a-64 basis 14695981039346656037 (0xCBF29CE484222325).
    prime        = 1099511628211  (0x100000001B3).
    fold: start h = offset basis; absorb each field's raw bytes little-endian
      at its full declared width (i8=1, i32=4, u64=8, two's complement); per
      byte b: h = (h XOR b) * prime (mod 2^64). No length prefix / terminator
      / final mix.

TOKEN FOLD (NORMATIVE, used by node signatures AND the committed kv_hash):
  for h_kv = 0..Hkv-1:
    fold k_scale_exp(h_kv)   (1 int8 byte)
    fold the D K bytes of h_kv  (d ascending)
    fold v_scale_exp(h_kv)   (1 int8 byte)
    fold the D V bytes of h_kv  (d ascending)

Input guarantees (harness invariants; implementations may rely on them):
  - draft_k/draft_v/bonus_k/bonus_v entries of real nodes are finite fp32,
    |x| <= 32768, and either 0.0f or |x| >= 2^-90 (every nonzero amax is a
    normal fp32). q entries are finite with |x| <= 4.
  - Padding node slots (i >= node_count[a]) contain arbitrary finite values
    and must not affect any output.
  - active_seq entries are unique within a step and in [0, B).
  - node_count entries are in [1, max_nodes].
  - parent[a][i] is -1 (root: attaches to the committed prefix) or in
    [0, i) (a valid earlier node of the same row's forest).
  - seq_len[b] + (max chain length + 1) never exceeds max_seq_len.
  - max_pages >= B * (ceil(max_seq_len / P) + ceil(max_nodes / P)): a page
    allocation never finds the pool empty in a conforming implementation.

Per-step RunSpec:
  active_count: rows in this step, 0 <= active_count <= B.
  step_id:      opaque, for debugging only.

Inputs (device pointers, N = max_nodes):
  active_seq[active_count]                 int32
  node_count[active_count]                 int32, in [1, N]
  parent[active_count, N]                  int32 (-1 or < node index)
  draft_k[active_count, N, Hkv, D]         fp32 (pad slots ignored)
  draft_v[active_count, N, Hkv, D]         fp32
  q[active_count, N, Hq, D]                fp32 (per-node queries)
  target_sig[active_count, N]              uint64
  bonus_k[active_count, Hkv, D]            fp32
  bonus_v[active_count, Hkv, D]            fp32

Step semantics (NORMATIVE order):
  (1) step_counter += 1.
  (2) Scratch allocation: for rows a = 0..active_count-1 ascending, allocate
      ceil(node_count[a] / P) scratch pages one at a time, each the FREE
      page with the LOWEST id; total_allocs increments per page. Node i
      lives at scratch page i/P, slot i%P of its row.
  (3) Quantize every real node's draft_k/draft_v (all Hkv heads) into its
      scratch slot per the storage format. Quantize each row's bonus token
      the same way (its bytes are stored on commit; where it is staged
      before commit is unobservable).
  (4) Attention: for every row a, real node i, query head hq, with
      L = seq_len[b] (committed, pre-commit) and the ancestor path
      P(i) = (root, ..., parent(i), i) obtained by following parent links:
        the attended token set is committed positions t = 0..L-1 (dequantized
        from the committed cache) followed by the path nodes (dequantized
        from scratch), in that order.
        score(token) = dot(q[a,i,hq,:], dequant_K(token, kvh, :)) * rsqrt(D)
        m = max score, l = sum exp(score - m)
        y[a,i,hq,d] = sum (exp(score-m)/l) * dequant_V(token, kvh, d)
        lse[a,i,hq] = m + log(l)
      The attended set always includes node i itself, so it is never empty
      even when L == 0. Any numerically stable evaluation order is allowed.
      Grader tolerance: |got - expected| <= ATOL + RTOL * |expected| with
      ATOL = RTOL = 2.5e-3 for both y and lse.
  (5) Verification (exact, per row):
        sig(i)  = FNV over node i's quantized scratch bytes (TOKEN FOLD).
        acc(i)  = (sig(i) == target_sig[a][i]) AND
                  (parent[i] == -1 OR acc(parent[i])).
        accepted_tail[a] = the accepted node with the greatest path depth
                  (root depth 1); ties broken by SMALLEST node index; -1 if
                  no node is accepted.
        accepted_len[a]  = path depth of accepted_tail (0 if none).
  (6) Commit: for rows a = 0..active_count-1 ascending: append the accepted
      path's tokens root-to-tail, then the bonus token, to the committed
      cache (seq_len[b] += accepted_len[a] + 1). Committed bytes are the
      EXACT scratch bytes (bit-identical; commit never requantizes). Each
      token appended at pos allocates committed page pos/P (lowest free id,
      total_allocs += 1) if absent. Scratch pages are still held during ALL
      commits (their ids are not reusable until phase 7).
  (7) Scratch release: all scratch pages of this step are freed
      (total_frees += 1 each). They are allocatable from the next step on.
  (8) Outputs (post-commit state). Pad slots (i >= node_count[a]) of y and
      lse must be written as exact 0.0f. All outputs are written on EVERY
      step, including active_count == 0.

Outputs (device pointers):
  y[active_count, N, Hq, D]     fp32   (tolerance above; pad slots exact 0)
  lse[active_count, N, Hq]      fp32   (tolerance above; pad slots exact 0)
  accepted_tail[active_count]   int32  exact (-1 if none)
  accepted_len[active_count]    int32  exact
  seq_len[B]                    int32  exact
  kv_hash[B]                    uint64 exact; per sequence b:
      h = basis
      for each committed token t = 0..seq_len[b]-1 ascending: TOKEN FOLD
      fold seq_len[b] (int32)
      (seq_len folds LAST: a running per-sequence hash over the append-only
      committed stream can be maintained incrementally)
  page_state_checksum[1]        uint64 exact; global:
      h = basis
      fold as int32 each: B, Hq, Hkv, D, page_size, max_nodes, max_seq_len,
                          max_pages, step_counter
      for b = 0..B-1 ascending:
        fold seq_len[b] (int32)
        n_lp = ceil(seq_len[b] / P); fold n_lp (int32)
        for lp = 0..n_lp-1: fold committed page_table[b][lp] (int32)
      fold total_allocs, total_frees, free_pages (each int32)
  free_pages[1]                 int32  exact (post-release)
  total_allocs[1]               int32  exact (cumulative since reset)
  total_frees[1]                int32  exact (cumulative since reset)

Determinism:
  - Exact replay of the same step sequence must reproduce every output bit
    (checksums included) exactly.
  - Permuting the row order within a step must reproduce y / lse /
    accepted_tail / accepted_len / seq_len / kv_hash / free_pages /
    total_allocs / total_frees exactly (page_state_checksum may differ:
    allocation interleaving is row-ordered by contract).

Rules for solution_run:
  - May launch kernels and use cudaMemsetAsync / device-to-device async copies.
  - May not call cudaMalloc/cudaFree (allocate persistent state in
    solution_init; persistent allocations up to ~1 GiB are acceptable).
  - May not perform synchronous host/device copies or otherwise synchronize
    the stream inside solution_run.
  - May not use CUB, Thrust, cuBLAS, cuDNN, or CUTLASS.
  - Must use only the provided workspace plus persistent state.
*/

struct alignas(8) StvProblemSpec {
    int32_t abi_version;
    int32_t B;
    int32_t Hq;
    int32_t Hkv;
    int32_t D;
    int32_t page_size;
    int32_t max_nodes;
    int32_t max_seq_len;
    int32_t max_pages;
    int32_t flags;
    int32_t reserved[6];
};

struct alignas(8) StvRunSpec {
    int32_t abi_version;
    int32_t active_count;
    int32_t step_id;
    int32_t reserved[13];
};

struct alignas(8) StvInputs {
    const int32_t* active_seq;
    const int32_t* node_count;
    const int32_t* parent;
    const float* draft_k;
    const float* draft_v;
    const float* q;
    const uint64_t* target_sig;
    const float* bonus_k;
    const float* bonus_v;
};

struct alignas(8) StvOutputs {
    float* y;
    float* lse;
    int32_t* accepted_tail;
    int32_t* accepted_len;
    int32_t* seq_len;
    uint64_t* kv_hash;
    uint64_t* page_state_checksum;
    int32_t* free_pages;
    int32_t* total_allocs;
    int32_t* total_frees;
};

static_assert(sizeof(StvProblemSpec) == 64, "StvProblemSpec layout drift");
static_assert(sizeof(StvRunSpec) == 64, "StvRunSpec layout drift");
static_assert(sizeof(StvInputs) == 72, "StvInputs layout drift");
static_assert(sizeof(StvOutputs) == 80, "StvOutputs layout drift");

static_assert(offsetof(StvProblemSpec, abi_version) == 0, "layout");
static_assert(offsetof(StvProblemSpec, B) == 4, "layout");
static_assert(offsetof(StvProblemSpec, Hq) == 8, "layout");
static_assert(offsetof(StvProblemSpec, Hkv) == 12, "layout");
static_assert(offsetof(StvProblemSpec, D) == 16, "layout");
static_assert(offsetof(StvProblemSpec, page_size) == 20, "layout");
static_assert(offsetof(StvProblemSpec, max_nodes) == 24, "layout");
static_assert(offsetof(StvProblemSpec, max_seq_len) == 28, "layout");
static_assert(offsetof(StvProblemSpec, max_pages) == 32, "layout");

static_assert(offsetof(StvRunSpec, abi_version) == 0, "layout");
static_assert(offsetof(StvRunSpec, active_count) == 4, "layout");
static_assert(offsetof(StvRunSpec, step_id) == 8, "layout");

static inline size_t stv_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int stv_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int stv_valid_D(int D) {
    return D == 64 || D == 128;
}

static inline int stv_valid_page_size(int P) {
    return P == 8 || P == 16 || P == 32;
}

static inline int stv_valid_max_nodes(int N) {
    return N == 16 || N == 32 || N == 64;
}

static inline int stv_max_logical_pages(int max_seq_len, int P) {
    return (max_seq_len + P - 1) / P;
}

static inline int stv_validate_problem_spec(const StvProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != STV_ABI_VERSION) return 0;
    if (spec->B < STV_MIN_B || spec->B > STV_MAX_B) return 0;
    if (spec->Hq < STV_MIN_HQ || spec->Hq > STV_MAX_HQ) return 0;
    if (spec->Hkv < STV_MIN_HKV || spec->Hkv > STV_MAX_HKV) return 0;
    if (spec->Hq % spec->Hkv != 0) return 0;
    if (!stv_valid_D(spec->D)) return 0;
    if (!stv_valid_page_size(spec->page_size)) return 0;
    if (!stv_valid_max_nodes(spec->max_nodes)) return 0;
    if (spec->max_seq_len < STV_MIN_SEQ_CAP || spec->max_seq_len > STV_MAX_SEQ_CAP) return 0;
    if (spec->max_pages < STV_MIN_PAGES || spec->max_pages > STV_MAX_PAGES) return 0;
    return 1;
}

static inline int stv_validate_run_spec(const StvRunSpec* run, const StvProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != STV_ABI_VERSION) return 0;
    if (run->active_count < 0 || run->active_count > spec->B) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const StvProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const StvProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const StvRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // SPEC_TREE_VERIFY_E5M2_KV_COMMON_H_
