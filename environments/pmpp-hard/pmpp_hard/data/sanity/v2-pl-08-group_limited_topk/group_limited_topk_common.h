// ============================================================================
// file: group_limited_topk_common.h
// ============================================================================

#ifndef GROUP_LIMITED_TOPK_COMMON_H_
#define GROUP_LIMITED_TOPK_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define GLT_ABI_VERSION 1

#define GLT_MIN_B 128
#define GLT_MAX_B 4096
#define GLT_MIN_V 256
#define GLT_MAX_V 4096
#define GLT_MIN_G 8
#define GLT_MAX_G 32

#define GLT_MAX_GROUP_K 4
#define GLT_MAX_N_GROUPS 8
#define GLT_MAX_FINAL_K 64

#define GLT_WEIGHT_ATOL 5.0e-5f
#define GLT_WEIGHT_RTOL 5.0e-5f
#define GLT_GROUP_SCORE_ATOL 1.0e-4f
#define GLT_GROUP_SCORE_RTOL 1.0e-4f

enum GltDistributionId : int32_t {
    GLT_DIST_UNIFORM = 0,
    GLT_DIST_PEAKED = 1,
    GLT_DIST_GROUP_SKEW = 2,
    GLT_DIST_MANY_TIES = 3
};

/*
CONTRACT: group_limited_topk

Stateless DeepSeek-style hierarchical group-limited top-k pipeline.
solution_init may be a no-op. solution_run handles B independent rows.

Legal shape family:
  B ∈ [128, 4096]
  V ∈ {256, 1024, 4096}
  G ∈ {8, 16, 32}
  V % G == 0
  group_size = V / G
  group_k ∈ {2, 4}
  n_groups ∈ {2, 4, 8}, with n_groups <= G
  final_k ∈ {8, 16, 64}

Inputs:
  score[B, V]:
    fp32 expert scores.

  uniform_u[B]:
    fp32 in [0, 1). Reserved for downstream deterministic sampling variants.
    This v1 grades hierarchical selection and weights only; uniform_u is not
    consumed by the required outputs.

Per-row pipeline:
  1. Within each group g, select top group_k experts by:
       larger score first,
       tie broken by smaller global expert id.
     group_k is clamped to group_size for completeness.

  2. group_score[g] =
       sum of the selected top group_k scores for that group.
     The selected experts used to form group_score are not separately output.

  3. Select top n_groups groups by:
       larger group_score first,
       tie broken by smaller group id.

  4. selected_group_ids[row, 0:n_groups]:
       selected group ids in that sorted order.

  5. Mask out all experts not belonging to selected groups.

  6. Over experts in selected groups, select final top-k by:
       larger score first,
       tie broken by smaller global expert id.

     survivor_count[row] =
       min(final_k, n_groups * group_size)

  7. survivor_expert_ids[row, 0:survivor_count]:
       expert ids in sorted final top-k order.

  8. weights[row, 0:survivor_count]:
       softmax over the selected final top-k scores, in the same order.
       Stable max-subtract is used.

Outputs:
  selected_group_ids[B, n_groups]:
    exact selected group ids.

  survivor_expert_ids[B, final_k]:
    exact selected expert ids for entries [0, survivor_count[row]).
    Entries beyond survivor_count are undefined and not graded.

  survivor_count[B]:
    exact number of selected experts.

  weights[B, final_k]:
    fp32 softmax weights for entries [0, survivor_count[row]).
    Compared with:
      abs_error <= 5e-5 + 5e-5 * abs(expected)

  group_scores[B, G]:
    fp32 group scores.
    Compared with:
      abs_error <= 1e-4 + 1e-4 * abs(expected)

Degenerate rules:
  - Ties inside a group use smaller global expert id.
  - Ties among groups use smaller group id.
  - Ties in final top-k use smaller global expert id.
  - If final_k > n_groups * group_size, all experts in selected groups survive.
  - Uniform/equal scores are therefore fully deterministic.

Rules for solution_run:
  - May launch kernels and use cudaMemsetAsync / device-to-device copies.
  - May not call cudaMalloc/cudaFree.
  - May not perform synchronous host/device copies.
  - May not use CUB, Thrust, cuBLAS, cuDNN, or CUTLASS.
  - Must use only the provided workspace plus persistent state allocated in
    solution_init.
*/

struct alignas(8) GltProblemSpec {
    int32_t abi_version;
    int32_t max_B;
    int32_t max_V;
    int32_t max_G;
    int32_t max_n_groups;
    int32_t max_final_k;
    int32_t flags;
    int32_t reserved[9];
};

struct alignas(8) GltRunSpec {
    int32_t abi_version;
    int32_t B;
    int32_t V;
    int32_t G;
    int32_t group_k;
    int32_t n_groups;
    int32_t final_k;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[6];
};

struct alignas(8) GltInputs {
    const float* score;
    const float* uniform_u;
};

struct alignas(8) GltOutputs {
    int32_t* selected_group_ids;
    int32_t* survivor_expert_ids;
    int32_t* survivor_count;
    float* weights;
    float* group_scores;
};

static_assert(sizeof(GltProblemSpec) == 64, "GltProblemSpec layout drift");
static_assert(sizeof(GltRunSpec) == 64, "GltRunSpec layout drift");
static_assert(sizeof(GltInputs) == 16, "GltInputs layout drift");
static_assert(sizeof(GltOutputs) == 40, "GltOutputs layout drift");

static_assert(offsetof(GltProblemSpec, abi_version) == 0, "layout");
static_assert(offsetof(GltProblemSpec, max_B) == 4, "layout");
static_assert(offsetof(GltProblemSpec, max_V) == 8, "layout");
static_assert(offsetof(GltProblemSpec, max_G) == 12, "layout");
static_assert(offsetof(GltProblemSpec, max_n_groups) == 16, "layout");
static_assert(offsetof(GltProblemSpec, max_final_k) == 20, "layout");
static_assert(offsetof(GltProblemSpec, flags) == 24, "layout");

static_assert(offsetof(GltRunSpec, abi_version) == 0, "layout");
static_assert(offsetof(GltRunSpec, B) == 4, "layout");
static_assert(offsetof(GltRunSpec, V) == 8, "layout");
static_assert(offsetof(GltRunSpec, G) == 12, "layout");
static_assert(offsetof(GltRunSpec, group_k) == 16, "layout");
static_assert(offsetof(GltRunSpec, n_groups) == 20, "layout");
static_assert(offsetof(GltRunSpec, final_k) == 24, "layout");
static_assert(offsetof(GltRunSpec, seed_id) == 28, "layout");
static_assert(offsetof(GltRunSpec, distribution_id) == 32, "layout");
static_assert(offsetof(GltRunSpec, case_id) == 36, "layout");

static inline size_t glt_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int glt_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int glt_valid_V(int V) {
    return V == 256 || V == 1024 || V == 4096;
}

static inline int glt_valid_G(int G) {
    return G == 8 || G == 16 || G == 32;
}

static inline int glt_valid_group_k(int group_k) {
    return group_k == 2 || group_k == 4;
}

static inline int glt_valid_n_groups(int n_groups) {
    return n_groups == 2 || n_groups == 4 || n_groups == 8;
}

static inline int glt_valid_final_k(int final_k) {
    return final_k == 8 || final_k == 16 || final_k == 64;
}

static inline int glt_validate_problem_spec(const GltProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != GLT_ABI_VERSION) return 0;
    if (spec->max_B < GLT_MIN_B || spec->max_B > GLT_MAX_B) return 0;
    if (!glt_valid_V(spec->max_V)) return 0;
    if (!glt_valid_G(spec->max_G)) return 0;
    if (spec->max_n_groups < 2 || spec->max_n_groups > GLT_MAX_N_GROUPS) return 0;
    if (spec->max_final_k < 8 || spec->max_final_k > GLT_MAX_FINAL_K) return 0;
    return 1;
}

static inline int glt_validate_run_spec(const GltRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != GLT_ABI_VERSION) return 0;
    if (run->B < GLT_MIN_B || run->B > GLT_MAX_B) return 0;
    if (!glt_valid_V(run->V)) return 0;
    if (!glt_valid_G(run->G)) return 0;
    if (run->V % run->G != 0) return 0;
    if (!glt_valid_group_k(run->group_k)) return 0;
    if (!glt_valid_n_groups(run->n_groups)) return 0;
    if (run->n_groups > run->G) return 0;
    if (!glt_valid_final_k(run->final_k)) return 0;
    if (run->distribution_id < GLT_DIST_UNIFORM ||
        run->distribution_id > GLT_DIST_MANY_TIES) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const GltProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const GltProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const GltRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // GROUP_LIMITED_TOPK_COMMON_H_
