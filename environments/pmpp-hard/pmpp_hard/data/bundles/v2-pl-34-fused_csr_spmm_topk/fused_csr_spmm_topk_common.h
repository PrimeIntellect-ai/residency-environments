// file: fused_csr_spmm_topk_common.h

#ifndef FUSED_CSR_SPMM_TOPK_COMMON_H_
#define FUSED_CSR_SPMM_TOPK_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define FCST_ABI_VERSION 1

#define FCST_MIN_ROWS 1
#define FCST_MAX_ROWS 65536
#define FCST_MIN_K 1
#define FCST_MAX_K 131072
#define FCST_MIN_N 1
#define FCST_MAX_N 4096
#define FCST_MAX_NNZ 4194304
#define FCST_MAX_M 64
#define FCST_REF_THREADS 64

enum FcstDistributionId : int32_t {
    FCST_DIST_UNIFORM = 0,
    FCST_DIST_POWERLAW = 1,
    FCST_DIST_GRID = 2,
    FCST_DIST_MANY_TIES = 3,
    FCST_DIST_ROW_HOT = 4
};

/*
CONTRACT: fused_csr_spmm_topk

Inputs:
  CSR sparse matrix A with shape [rows, K]:
    row_offsets[rows + 1] int32
    col_indices[nnz]      int32
    vals[nnz]             int32

  Dense matrix B with shape [K, N]:
    dense_b[K, N] int32 row-major

For each sparse row r and dense column n:
  spmm_value[r,n] =
    int64 wrapping sum over e in [row_offsets[r], row_offsets[r+1]):
      if 0 <= col_indices[e] < K:
        vals[e] * dense_b[col_indices[e], n]
      else:
        ignored

Product int32*int32 fits int64. The reduction sum wraps mod 2^64 and is
interpreted as signed int64 for ordering and output.

For each row r:
  select TOP-M columns by:
    1. spmm_value[r,n] descending, signed int64 order
    2. smaller n first for equal values

Outputs:
  topm_cols[rows, M] int32:
    selected dense column ids, sorted by the above order

  topm_vals[rows, M] int64:
    selected spmm values aligned to topm_cols

  topm_count[rows] int32:
    min(M, N)

  row_sum[rows] int64:
    int64 wrapping sum of topm_vals over selected top-M entries

  row_max[rows] int64:
    maximum top-M value, or INT64_MIN if topm_count == 0

  row_argmax[rows] int32:
    first column id attaining row_max among top-M; tie smaller col.
    -1 if topm_count == 0

  row_nnz[rows] int32:
    row_offsets[r+1] - row_offsets[r]

  row_nnz_prefix[rows+1] int32:
    exact copy of row_offsets[0:rows+1]

Rules:
  - solution_run may not call cudaMalloc/cudaFree.
  - solution_run may launch kernels and use supplied workspace.
  - No CUB/Thrust.
  - All graded outputs are exact.
*/

struct alignas(8) FcstProblemSpec {
    int32_t abi_version;
    int32_t max_rows;
    int32_t max_K;
    int32_t max_N;
    int32_t max_nnz;
    int32_t max_M;
    int32_t flags;
    int32_t reserved[9];
};

struct alignas(8) FcstRunSpec {
    int32_t abi_version;
    int32_t rows;
    int32_t K;
    int32_t N;
    int32_t nnz;
    int32_t M;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[7];
};

struct alignas(8) FcstInputs {
    const int32_t* row_offsets;
    const int32_t* col_indices;
    const int32_t* vals;
    const int32_t* dense_b;
};

struct alignas(8) FcstOutputs {
    int32_t* topm_cols;
    int64_t* topm_vals;
    int32_t* topm_count;
    int64_t* row_sum;
    int64_t* row_max;
    int32_t* row_argmax;
    int32_t* row_nnz;
    int32_t* row_nnz_prefix;
};

static_assert(sizeof(FcstProblemSpec) == 64, "FcstProblemSpec layout drift");
static_assert(sizeof(FcstRunSpec) == 64, "FcstRunSpec layout drift");
static_assert(sizeof(FcstInputs) == 32, "FcstInputs layout drift");
static_assert(sizeof(FcstOutputs) == 64, "FcstOutputs layout drift");

static inline size_t fcst_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int fcst_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int fcst_validate_problem_spec(const FcstProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != FCST_ABI_VERSION) return 0;
    if (spec->max_rows < FCST_MIN_ROWS || spec->max_rows > FCST_MAX_ROWS) return 0;
    if (spec->max_K < FCST_MIN_K || spec->max_K > FCST_MAX_K) return 0;
    if (spec->max_N < FCST_MIN_N || spec->max_N > FCST_MAX_N) return 0;
    if (spec->max_nnz < 0 || spec->max_nnz > FCST_MAX_NNZ) return 0;
    if (spec->max_M < 1 || spec->max_M > FCST_MAX_M) return 0;
    return 1;
}

static inline int fcst_validate_run_spec(const FcstRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != FCST_ABI_VERSION) return 0;
    if (run->rows < FCST_MIN_ROWS || run->rows > FCST_MAX_ROWS) return 0;
    if (run->K < FCST_MIN_K || run->K > FCST_MAX_K) return 0;
    if (run->N < FCST_MIN_N || run->N > FCST_MAX_N) return 0;
    if (run->nnz < 0 || run->nnz > FCST_MAX_NNZ) return 0;
    if (run->M < 1 || run->M > FCST_MAX_M) return 0;
    if (run->distribution_id < FCST_DIST_UNIFORM ||
        run->distribution_id > FCST_DIST_ROW_HOT) return 0;
    return 1;
}

static inline int fcst_topm_count_host(int N, int M) {
    return M < N ? M : N;
}

extern "C" size_t solution_workspace_bytes(const FcstProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const FcstProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const FcstRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // FUSED_CSR_SPMM_TOPK_COMMON_H_
