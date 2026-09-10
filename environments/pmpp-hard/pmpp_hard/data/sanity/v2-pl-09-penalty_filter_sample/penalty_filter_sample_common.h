// ============================================================================
// file: penalty_filter_sample_common.h
// ============================================================================

#ifndef PENALTY_FILTER_SAMPLE_COMMON_H_
#define PENALTY_FILTER_SAMPLE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define PFS_ABI_VERSION 1

#define PFS_MIN_B 256
#define PFS_MAX_B 4096
#define PFS_MIN_V 1024
#define PFS_MAX_V 32768
#define PFS_MAX_HISTORY 256

#define PFS_PROB_ATOL 5.0e-5f
#define PFS_PROB_RTOL 5.0e-5f

enum PfsDistributionId : int32_t {
    PFS_DIST_UNIFORM = 0,
    PFS_DIST_PEAKED = 1,
    PFS_DIST_HEAVY_TAIL = 2,
    PFS_DIST_MANY_TIES = 3,
    PFS_DIST_ALL_EQUAL = 4
};

/*
CONTRACT: penalty_filter_sample

Stateless penalty + min-p sampler pipeline. solution_init may be a no-op.
solution_run handles B independent rows.

Legal shape family:
  B ∈ [256, 4096]
  V ∈ {1024, 4096, 32768}
  H ∈ [0, 256], the per-row history storage capacity

Inputs:
  logits[B, V]:
    fp32 logits.

  history_token[B, H]:
    int32 previous-token IDs for each row.
    Only the first history_len[row] entries are live.
    Invalid token IDs outside [0, V) are ignored.
    If H == 0, history_token is not read.

  history_len[B]:
    int32 live history length for each row.
    Sanitized as clamp(history_len[row], 0, H).

  repetition_penalty[B]:
    fp32 rp. Generated values are > 1.0.
    Sanitized as max(rp, 1.0).

  frequency_penalty[B]:
    fp32 fp.

  presence_penalty[B]:
    fp32 pp.

  temperature[B]:
    fp32 t. Generated values are > 0.
    Sanitized as max(t, 1e-6).

  min_p[B]:
    fp32 in [0, 1). Sanitized as clamp(min_p, 0, 0.99999994).

  uniform_u[B]:
    fp32 in [0, 1). Used for deterministic sampling after final
    renormalization. If u < 0, treat as 0. If u >= 1, treat as the largest
    representable value below 1.

Penalty formula per row/token:
  count_i = number of occurrences of token i in the live history.

  x = logits_i

  if count_i > 0:
    if x > 0: x = x / repetition_penalty
    else:     x = x * repetition_penalty

    x = x - frequency_penalty * count_i - presence_penalty

  If count_i == 0, no penalty is applied.

  s_i = x / temperature

Canonical ordering:
  The SAME ordering is used for min-p survivor packing and sampling:
    1. adjusted score s_i descending,
    2. tie broken by smaller token id.

Pipeline per row:
  1. Apply repetition/frequency/presence penalties using history counts.
  2. Divide by temperature.
  3. Conceptual full softmax over all V tokens using stable max-subtract.
  4. MIN-P filter:
       max_prob is the softmax probability of the first canonical token.
       Keep token i iff:
         prob_i >= min_p * max_prob

       Since softmax denominator cancels, implementations may equivalently use:
         exp(s_i - s_max) >= min_p

       min_p == 0 keeps all tokens.
       At least one token always survives.
  5. Final renormalization over survivors only:
       packed_cand_prob_i = exp(s_i - s_max_survivor) /
                            Σ_survivors exp(s_j - s_max_survivor)
  6. Deterministic sampling:
       Walk survivors in canonical order, accumulating packed_cand_prob in
       fp32 order. Select the FIRST token whose running cumulative probability
       is strictly greater than u. If no token satisfies this due to fp rounding,
       select the final survivor.

Outputs:
  selected_token[B]:
    exact deterministic sampled token id.

  survivor_count[B]:
    exact number of survivors after min-p.
    Always >= 1.

  packed_cand_token[B, V]:
    surviving token ids in canonical order.
    Only entries [row*V, row*V + survivor_count[row]) are defined/graded.

  packed_cand_prob[B, V]:
    final renormalized survivor weights in the same order.
    Only entries [row*V, row*V + survivor_count[row]) are defined/graded.
    Compared with:
      abs_error <= 5e-5 + 5e-5 * abs(expected)

Degenerate rules:
  - Empty history: no penalties.
  - min_p == 0: all V tokens survive.
  - all-equal adjusted scores: ordering is token id ascending.
  - all tokens penalized: the same rules apply after penalty.
  - min_p tiny: at least the first canonical token survives.

Rules for solution_run:
  - May launch kernels and use cudaMemsetAsync / device-to-device copies.
  - May not call cudaMalloc/cudaFree.
  - May not perform synchronous host/device copies.
  - May not use CUB, Thrust, cuBLAS, cuDNN, or CUTLASS.
  - Must use only the provided workspace plus persistent state allocated in
    solution_init.
*/

struct alignas(8) PfsProblemSpec {
    int32_t abi_version;
    int32_t max_B;
    int32_t max_V;
    int32_t max_H;
    int32_t flags;
    int32_t reserved[11];
};

struct alignas(8) PfsRunSpec {
    int32_t abi_version;
    int32_t B;
    int32_t V;
    int32_t H;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[9];
};

struct alignas(8) PfsInputs {
    const float* logits;
    const int32_t* history_token;
    const int32_t* history_len;
    const float* repetition_penalty;
    const float* frequency_penalty;
    const float* presence_penalty;
    const float* temperature;
    const float* min_p;
    const float* uniform_u;
};

struct alignas(8) PfsOutputs {
    int32_t* selected_token;
    int32_t* survivor_count;
    int32_t* packed_cand_token;
    float* packed_cand_prob;
};

static_assert(sizeof(PfsProblemSpec) == 64, "PfsProblemSpec layout drift");
static_assert(sizeof(PfsRunSpec) == 64, "PfsRunSpec layout drift");
static_assert(sizeof(PfsInputs) == 72, "PfsInputs layout drift");
static_assert(sizeof(PfsOutputs) == 32, "PfsOutputs layout drift");

static_assert(offsetof(PfsProblemSpec, abi_version) == 0, "layout");
static_assert(offsetof(PfsProblemSpec, max_B) == 4, "layout");
static_assert(offsetof(PfsProblemSpec, max_V) == 8, "layout");
static_assert(offsetof(PfsProblemSpec, max_H) == 12, "layout");
static_assert(offsetof(PfsProblemSpec, flags) == 16, "layout");

static_assert(offsetof(PfsRunSpec, abi_version) == 0, "layout");
static_assert(offsetof(PfsRunSpec, B) == 4, "layout");
static_assert(offsetof(PfsRunSpec, V) == 8, "layout");
static_assert(offsetof(PfsRunSpec, H) == 12, "layout");
static_assert(offsetof(PfsRunSpec, seed_id) == 16, "layout");
static_assert(offsetof(PfsRunSpec, distribution_id) == 20, "layout");
static_assert(offsetof(PfsRunSpec, case_id) == 24, "layout");

static inline size_t pfs_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int pfs_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int pfs_valid_V(int V) {
    return V == 1024 || V == 4096 || V == 32768;
}

static inline int pfs_validate_problem_spec(const PfsProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != PFS_ABI_VERSION) return 0;
    if (spec->max_B < PFS_MIN_B || spec->max_B > PFS_MAX_B) return 0;
    if (!pfs_valid_V(spec->max_V)) return 0;
    if (spec->max_H < 0 || spec->max_H > PFS_MAX_HISTORY) return 0;
    return 1;
}

static inline int pfs_validate_run_spec(const PfsRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != PFS_ABI_VERSION) return 0;
    if (run->B < PFS_MIN_B || run->B > PFS_MAX_B) return 0;
    if (!pfs_valid_V(run->V)) return 0;
    if (run->H < 0 || run->H > PFS_MAX_HISTORY) return 0;
    if (run->distribution_id < PFS_DIST_UNIFORM ||
        run->distribution_id > PFS_DIST_ALL_EQUAL) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const PfsProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const PfsProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const PfsRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // PENALTY_FILTER_SAMPLE_COMMON_H_
