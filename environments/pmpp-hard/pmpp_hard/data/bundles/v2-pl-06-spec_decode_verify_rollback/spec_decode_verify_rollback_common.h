// ============================================================================
// file: spec_decode_verify_rollback_common.h
// ============================================================================

#ifndef SPEC_DECODE_VERIFY_ROLLBACK_COMMON_H_
#define SPEC_DECODE_VERIFY_ROLLBACK_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define SDV_ABI_VERSION 1

#define SDV_MIN_B 1
#define SDV_MAX_B 64
#define SDV_MIN_MAX_LEN 16
#define SDV_MAX_MAX_LEN 4096
#define SDV_MIN_MAX_STEPS 16
#define SDV_MAX_MAX_STEPS 128
#define SDV_MAX_DRAFT_LEN 8
#define SDV_TAIL_VALUES 32

/*
CONTRACT: spec_decode_verify_rollback

This is a stateful exact-integer speculative-decoding verification task.
solution_init allocates persistent append-only per-sequence token caches,
lengths, and checkpoint lengths. solution_run advances one verification step.
solution_reset clears persistent state.

Legal configuration:
  B:
    number of logical sequences.
    1 <= B <= 64.

  max_len:
    per-sequence cache capacity.
    16 <= max_len <= 4096.

  max_steps:
    maximum number of solution_run calls between resets.
    16 <= max_steps <= 128.
    This exists so naive replay implementations can allocate bounded history.

Persistent state:
  cache[B, max_len]:
    int32 token/value cache.

  length[B]:
    current live cache length for each sequence.

  checkpoint_length[B]:
    saved pre-step length for the most recent active update of each sequence.
    Inactive sequences keep their prior checkpoint_length.

Per-step RunSpec:
  active_count:
    number of active rows in this step, 0 <= active_count <= B.

  draft_len:
    L, number of proposed draft tokens per active row.
    L ∈ {2, 4, 8}.

  step_id:
    opaque step identifier for grading/debugging only.

Inputs:
  active_seq[active_count]:
    logical sequence IDs in [0, B). Entries are unique within a step.

  draft_value[active_count, draft_len]:
    int32 proposed draft values. The implementation must tentatively append
    these values logically, then keep only the accepted prefix.

  correction_value[active_count]:
    int32 correction token appended after rollback.

  p_target[active_count, draft_len]:
    uint32 exact target probability weight.

  p_draft[active_count, draft_len]:
    uint32 exact draft probability weight.

  uniform_u32[active_count, draft_len]:
    uint32 exact uniform draw.

Acceptance rule:
  For draft token i:
    if p_draft_i == 0:
      accept_i = (p_target_i > 0)
    else:
      accept_i =
        uint64(uniform_u32_i) * uint64(p_draft_i)
          <=
        uint64(p_target_i) * 4294967295ULL

  The accepted prefix a is the number of leading accept_i == true values.
  Stop at the first rejected token.

Rollback/update semantics:
  For each active row in active_seq order:
    old_length = length[seq]
    checkpoint_length[seq] = old_length

    Compute accepted prefix a over all draft_len bits.

    Logically:
      tentatively append all L draft values,
      rollback to old_length + a,
      append one correction token.

    Capacity cap rule:
      The physical live cache is capped at max_len.
      retained_accepted =
        min(a, max(0, max_len - old_length))

      The correction token is appended iff:
        old_length + a < max_len

      new_length =
        min(max_len, old_length + a + 1)

      Draft values retained in cache are:
        draft_value[0 : retained_accepted)

      The correction value is written at index old_length + a only if that
      index is < max_len.

      Values beyond new_length are dead/stale and must not contribute to
      outputs or checksums. They do not need to be cleared.

Outputs:
  accepted_count[active_count]:
    number of accepted draft tokens physically retained in the live cache after
    applying the capacity cap. This may be less than the logical prefix a when
    the cache is near full.

  new_length[active_count]:
    updated live length for that active sequence.

  live_cache_sum[active_count]:
    int64 two's-complement wrapping sum over cache[seq, 0:new_length).

  live_cache_tail_hash[active_count]:
    FNV-1a-64 PRIMITIVE (NORMATIVE -- this project uses a NON-CANONICAL basis;
      the agent MUST use this exact basis or every checksum fails):
        offset basis = 1469598103934665603  (hex 0x14650FB0739D0383) -- this is NOT
          the canonical FNV-1a-64 basis 14695981039346656037 (0xCBF29CE484222325).
        prime        = 1099511628211  (0x100000001B3).
        fold: start h = offset basis; absorb each field's raw bytes little-endian at
          its full declared width (u8=1, u32=4, u64=8, i64=8 two's-complement); per
          byte b: h = (h XOR b) * prime (mod 2^64). No length prefix / terminator / final mix.

    FNV-1a 64-bit hash over:
      tail_count int32 bytes,
      then the last tail_count int32 cache values in live order,
    where tail_count = min(new_length, 32).

  state_checksum[1]:
    FNV-1a 64-bit checksum over persistent state in this exact order:
      length[B] int32 bytes
      checkpoint_length[B] int32 bytes
      for b in [0, B):
        live_len = length[b] int32 bytes
        cache[b, 0:live_len] int32 bytes

Rules for solution_run:
  - May launch kernels and use cudaMemsetAsync / device-to-device copies.
  - May not call cudaMalloc/cudaFree.
  - May not perform synchronous host/device copies.
  - May not use CUB, Thrust, cuBLAS, cuDNN, or CUTLASS.
  - Must use only the provided workspace plus persistent state allocated in
    solution_init.
*/

struct alignas(8) SdvProblemSpec {
    int32_t abi_version;
    int32_t B;
    int32_t max_len;
    int32_t max_steps;
    int32_t flags;
    int32_t reserved[11];
};

struct alignas(8) SdvRunSpec {
    int32_t abi_version;
    int32_t active_count;
    int32_t draft_len;
    int32_t step_id;
    int32_t reserved[12];
};

struct alignas(8) SdvInputs {
    const int32_t* active_seq;
    const int32_t* draft_value;
    const int32_t* correction_value;
    const uint32_t* p_target;
    const uint32_t* p_draft;
    const uint32_t* uniform_u32;
};

struct alignas(8) SdvOutputs {
    int32_t* accepted_count;
    int32_t* new_length;
    int64_t* live_cache_sum;
    uint64_t* live_cache_tail_hash;
    uint64_t* state_checksum;
};

static_assert(sizeof(SdvProblemSpec) == 64, "SdvProblemSpec layout drift");
static_assert(sizeof(SdvRunSpec) == 64, "SdvRunSpec layout drift");
static_assert(sizeof(SdvInputs) == 48, "SdvInputs layout drift");
static_assert(sizeof(SdvOutputs) == 40, "SdvOutputs layout drift");

static_assert(offsetof(SdvProblemSpec, abi_version) == 0, "layout");
static_assert(offsetof(SdvProblemSpec, B) == 4, "layout");
static_assert(offsetof(SdvProblemSpec, max_len) == 8, "layout");
static_assert(offsetof(SdvProblemSpec, max_steps) == 12, "layout");
static_assert(offsetof(SdvProblemSpec, flags) == 16, "layout");

static_assert(offsetof(SdvRunSpec, abi_version) == 0, "layout");
static_assert(offsetof(SdvRunSpec, active_count) == 4, "layout");
static_assert(offsetof(SdvRunSpec, draft_len) == 8, "layout");
static_assert(offsetof(SdvRunSpec, step_id) == 12, "layout");

static inline size_t sdv_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int sdv_valid_draft_len(int L) {
    return L == 2 || L == 4 || L == 8;
}

static inline int sdv_validate_problem_spec(const SdvProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != SDV_ABI_VERSION) return 0;
    if (spec->B < SDV_MIN_B || spec->B > SDV_MAX_B) return 0;
    if (spec->max_len < SDV_MIN_MAX_LEN || spec->max_len > SDV_MAX_MAX_LEN) return 0;
    if (spec->max_steps < SDV_MIN_MAX_STEPS || spec->max_steps > SDV_MAX_MAX_STEPS) return 0;
    return 1;
}

static inline int sdv_validate_run_spec(const SdvRunSpec* run, const SdvProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != SDV_ABI_VERSION) return 0;
    if (run->active_count < 0 || run->active_count > spec->B) return 0;
    if (!sdv_valid_draft_len(run->draft_len)) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const SdvProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const SdvProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const SdvRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // SPEC_DECODE_VERIFY_ROLLBACK_COMMON_H_
