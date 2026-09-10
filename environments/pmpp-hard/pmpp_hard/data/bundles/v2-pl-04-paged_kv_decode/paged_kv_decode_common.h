// ============================================================================
// file: paged_kv_decode_common.h
// ============================================================================

#ifndef PAGED_KV_DECODE_COMMON_H_
#define PAGED_KV_DECODE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define PKD_ABI_VERSION 1

#define PKD_MIN_B 1
#define PKD_MAX_B 64
#define PKD_MIN_HQ 1
#define PKD_MAX_HQ 32
#define PKD_MIN_HKV 1
#define PKD_MAX_HKV 16
#define PKD_MIN_D 64
#define PKD_MAX_D 128
#define PKD_MIN_PAGE_SIZE 16
#define PKD_MAX_PAGE_SIZE 32
#define PKD_MIN_SEQ_LEN 1
#define PKD_MAX_SEQ_LEN 2048
#define PKD_MAX_PAGES 8192
#define PKD_MAX_NEW_TOKENS 2

#define PKD_Y_ATOL 2.0e-3f
#define PKD_Y_RTOL 2.0e-3f

/*
CONTRACT: paged_kv_decode

This is a stateful single-GPU decode pipeline. solution_init allocates a
persistent paged int8 KV cache, page scales, page tables, sequence lengths, and
a page allocator. solution_run advances exactly one decode step. solution_reset
clears all persistent state.

Legal configuration:
  B:
    batch size / number of logical sequences.
    1 <= B <= 64.

  Hq:
    number of query heads.
    1 <= Hq <= 32.

  Hkv:
    number of KV heads.
    1 <= Hkv <= 16.
    Hq % Hkv == 0.

  GQA group:
    group = Hq / Hkv.
    Query head hq reads KV head hq / group.

  D:
    head dimension.
    D ∈ {64, 128}.

  page_size P:
    P ∈ {16, 32}.

  max_seq_len:
    maximum sequence length for every logical sequence.
    1 <= max_seq_len <= 2048.

  max_pages:
    number of physical pages allocated in persistent state.
    Must be >= B * ceil(max_seq_len / P).
    Must be <= 8192.

Persistent state:
  page_table[B, pages_per_seq]:
    int32 physical page ID for each logical sequence page slot.
    pages_per_seq = ceil(max_seq_len / P).
    Unallocated entries are -1.

  lengths[B]:
    current logical sequence lengths.

  next_page:
    monotonically increasing physical-page allocator.

  k_cache[max_pages, Hkv, P, D]:
    int8 quantized K cache.

  v_cache[max_pages, Hkv, P, D]:
    int8 quantized V cache.

  page_scale[max_pages, Hkv]:
    fp32 scale for both K and V within that physical page and KV head.

Per-step RunSpec:
  active_count:
    number of active rows in this step, 0 <= active_count <= B.

  step_id:
    opaque monotonic step identifier for grading/debugging only.

  window_size:
    sliding-window length W.
    Attention range for each active sequence is:
      [max(0, length_after_append - W), length_after_append)
    If W > max_seq_len, it is clamped to max_seq_len.
    W must be positive in generated tests.

Inputs:
  active_seq[active_count]:
    logical sequence IDs in [0, B). Entries are unique within the step.

  new_token_count[active_count]:
    number of new KV tokens to append for that active sequence.
    Each value is in {0, 1, 2}.

  new_k[active_count, 2, Hkv, D]:
    fp32 K vectors for the up-to-two appended tokens. Entries for unused
    new-token slots are ignored.

  new_v[active_count, 2, Hkv, D]:
    fp32 V vectors for the up-to-two appended tokens.

  new_scale[active_count, 2, Hkv]:
    PROVIDED per-page scale candidates.
    If an appended token starts a new physical page, then for every KV head h:
      page_scale[new_page, h] = sanitize(new_scale[a, nt, h])
    where sanitize(s) = s if s > 1e-6f, otherwise 1.0f.
    If an appended token lands in an already allocated page, new_scale is
    ignored and the existing page_scale is reused.

  q[active_count, Hq, D]:
    fp32 query vectors for the current decode step.

Append and quantization:
  Appends happen for active rows in active_seq order. Page allocation is
  deterministic in that same order.

  FULL-SEQUENCE EDGE RULE (normative): if a sequence is already at capacity
  (old_len = lengths[seq] >= max_seq_len) when its append would occur, the append
  is SILENTLY SKIPPED for that token: no page is allocated, length is unchanged,
  and that entry's append_pos and append_page outputs are set to -1. Subsequent
  active rows still process normally.

  Each appended K/V value x is quantized as:
    z = x / scale
    q = round_half_away_from_zero(z)
    q = clamp(q, -127, 127)
    int8(q)

  Dequantization uses:
    fp32(q_int8) * scale

Attention:
  After all active appends for the step have completed, compute one output row
  for every active sequence and query head:

    y[a, hq, d] =
      softmax_j(score_j) weighted sum over V[j, kvh, d]

    kvh = hq / (Hq / Hkv)
    score_j = dot(q[a,hq,:], dequant(K[token_j,kvh,:])) * rsqrt(D)

  The attention range is the sequence's cached tokens within the causal
  sliding window:
    j in [max(0, length_after_append - W), length_after_append)

  If the resulting range is empty, y[a,hq,d] = 0.

  Implementations may use any numerically stable softmax. The grader compares
  y with:
    abs_error <= 2e-3 + 2e-3 * abs(expected)

Outputs:
  y[active_count, Hq, D]:
    fp32 attention output.

  lengths[B]:
    updated lengths after the step.

  state_checksum[1]:
    FNV-1a-64 PRIMITIVE (NORMATIVE -- this project uses a NON-CANONICAL basis;
      the agent MUST use this exact basis or every checksum fails):
        offset basis = 1469598103934665603  (hex 0x14650FB0739D0383) -- this is NOT
          the canonical FNV-1a-64 basis 14695981039346656037 (0xCBF29CE484222325).
        prime        = 1099511628211  (0x100000001B3).
        fold: start h = offset basis; absorb each field's raw bytes little-endian at
          its full declared width (u8=1, u32=4, u64=8, i64=8 two's-complement); per
          byte b: h = (h XOR b) * prime (mod 2^64). No length prefix / terminator / final mix.

    FNV-1a 64-bit checksum over persistent state, in this exact order:
      next_page int32 bytes
      lengths[B] int32 bytes
      page_table[B * pages_per_seq] int32 bytes
      for p in [0, next_page):
        page_scale[p, Hkv] float bytes
        k_cache[p, Hkv, P, D] int8 bytes
        v_cache[p, Hkv, P, D] int8 bytes
    This checksum is exact and is used to catch stale page-table/cache bugs.

Rules for solution_run:
  - May launch kernels and use cudaMemsetAsync / device-to-device copies.
  - May not call cudaMalloc/cudaFree.
  - May not perform synchronous host/device copies.
  - May not use CUB, Thrust, cuBLAS, cuDNN, or CUTLASS.
  - Must use only the provided workspace plus persistent state allocated in
    solution_init.
*/

struct alignas(8) PkdProblemSpec {
    int32_t abi_version;
    int32_t B;
    int32_t Hq;
    int32_t Hkv;
    int32_t D;
    int32_t page_size;
    int32_t max_seq_len;
    int32_t max_pages;
    int32_t flags;
    int32_t reserved[7];
};

struct alignas(8) PkdRunSpec {
    int32_t abi_version;
    int32_t active_count;
    int32_t step_id;
    int32_t window_size;
    int32_t reserved[12];
};

struct alignas(8) PkdInputs {
    const int32_t* active_seq;
    const int32_t* new_token_count;
    const float* new_k;
    const float* new_v;
    const float* new_scale;
    const float* q;
};

struct alignas(8) PkdOutputs {
    float* y;
    int32_t* lengths;
    uint64_t* state_checksum;
};

static_assert(sizeof(PkdProblemSpec) == 64, "PkdProblemSpec layout drift");
static_assert(sizeof(PkdRunSpec) == 64, "PkdRunSpec layout drift");
static_assert(sizeof(PkdInputs) == 48, "PkdInputs layout drift");
static_assert(sizeof(PkdOutputs) == 24, "PkdOutputs layout drift");

static_assert(offsetof(PkdProblemSpec, abi_version) == 0, "layout");
static_assert(offsetof(PkdProblemSpec, B) == 4, "layout");
static_assert(offsetof(PkdProblemSpec, Hq) == 8, "layout");
static_assert(offsetof(PkdProblemSpec, Hkv) == 12, "layout");
static_assert(offsetof(PkdProblemSpec, D) == 16, "layout");
static_assert(offsetof(PkdProblemSpec, page_size) == 20, "layout");
static_assert(offsetof(PkdProblemSpec, max_seq_len) == 24, "layout");
static_assert(offsetof(PkdProblemSpec, max_pages) == 28, "layout");
static_assert(offsetof(PkdProblemSpec, flags) == 32, "layout");

static_assert(offsetof(PkdRunSpec, abi_version) == 0, "layout");
static_assert(offsetof(PkdRunSpec, active_count) == 4, "layout");
static_assert(offsetof(PkdRunSpec, step_id) == 8, "layout");
static_assert(offsetof(PkdRunSpec, window_size) == 12, "layout");

static inline size_t pkd_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int pkd_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int pkd_valid_D(int D) {
    return D == 64 || D == 128;
}

static inline int pkd_valid_page_size(int P) {
    return P == 16 || P == 32;
}

static inline int pkd_pages_per_seq(int max_seq_len, int page_size) {
    return pkd_ceil_div_int(max_seq_len, page_size);
}

static inline int pkd_validate_problem_spec(const PkdProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != PKD_ABI_VERSION) return 0;
    if (spec->B < PKD_MIN_B || spec->B > PKD_MAX_B) return 0;
    if (spec->Hq < PKD_MIN_HQ || spec->Hq > PKD_MAX_HQ) return 0;
    if (spec->Hkv < PKD_MIN_HKV || spec->Hkv > PKD_MAX_HKV) return 0;
    if (spec->Hq % spec->Hkv != 0) return 0;
    if (!pkd_valid_D(spec->D)) return 0;
    if (!pkd_valid_page_size(spec->page_size)) return 0;
    if (spec->max_seq_len < PKD_MIN_SEQ_LEN || spec->max_seq_len > PKD_MAX_SEQ_LEN) return 0;
    if (spec->max_pages < 1 || spec->max_pages > PKD_MAX_PAGES) return 0;

    const int pages_per_seq = pkd_pages_per_seq(spec->max_seq_len, spec->page_size);
    if (spec->max_pages < spec->B * pages_per_seq) return 0;

    return 1;
}

static inline int pkd_validate_run_spec(const PkdRunSpec* run, const PkdProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != PKD_ABI_VERSION) return 0;
    if (run->active_count < 0 || run->active_count > spec->B) return 0;
    if (run->window_size <= 0) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const PkdProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const PkdProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const PkdRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // PAGED_KV_DECODE_COMMON_H_
