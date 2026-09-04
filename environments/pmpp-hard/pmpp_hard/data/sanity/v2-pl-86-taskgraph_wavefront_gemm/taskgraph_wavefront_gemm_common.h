// ============================================================================
// file: taskgraph_wavefront_gemm_common.h
// ============================================================================

#ifndef TASKGRAPH_WAVEFRONT_GEMM_COMMON_H_
#define TASKGRAPH_WAVEFRONT_GEMM_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define TWG_ABI_VERSION 1

// Tile edge (fixed).
#define TWG_T 32

#define TWG_MIN_R 1
#define TWG_MAX_R 32
#define TWG_MIN_C 1
#define TWG_MAX_C 32
#define TWG_MIN_K 64
#define TWG_MAX_K 256
#define TWG_MIN_P 0
#define TWG_MAX_P 15

// Salt mixing constants (normative).
#define TWG_SALT_ROW 0x9E3779B9u
#define TWG_SALT_COL 0x85EBCA6Bu
#define TWG_SALT_ROUND 0xC2B2AE35u

// FNV-1a-64 (NON-canonical basis used across this project).
#define TWG_FNV_BASIS 1469598103934665603ULL
#define TWG_FNV_PRIME 1099511628211ULL

enum TwgDistributionId : int32_t {
    TWG_DIST_UNIFORM = 0,
    TWG_DIST_SMALL = 1,
    TWG_DIST_SPARSE = 2,
    TWG_DIST_SATURATE = 3,
    TWG_DIST_ZERO_U = 4
};

/*
CONTRACT: taskgraph_wavefront_gemm

A stateful GPU-resident mini-system. Persistent state is an R x C grid of
32x32 uint32 accumulator tiles ACC (plus a round counter). Each solution_run
executes ONE round: every tile (i, j) is updated exactly once by a task whose
inputs include the ALREADY-UPDATED left and top neighbor tiles of the SAME
round -- a wavefront dependency DAG. The per-task math is a salted int8 tile
GEMM whose operand bytes are XOR-mixed with a salt derived from the updated
neighbor tiles, so a task cannot start its heavy arithmetic until both
neighbors are complete: the schedule is real, and there is no algebraic
shortcut that decouples the DAG. All arithmetic is exact (uint32 wraparound
two's-complement); there is no tolerance anywhere.

Legal shape family (RunSpec):
  R in [1, 32], C in [1, 32]       (tile grid; TWG_T = 32 fixed)
  K in {64, 128, 256}              (GEMM depth; may differ between rounds)
  P1, P2 in [0, 15]                (linear coupling coefficients)
  dump in {0, 1}                   (1 => also export the full ACC state)

State lifecycle:
  solution_init : allocate persistent state for the ProblemSpec maxima.
  solution_reset: ACC := all zeros, round counter := 0.
  solution_run  : execute one round (see below), round counter += 1.
  The grader calls solution_reset before changing R or C; between resets all
  runs use the same (R, C). K, P1, P2, dump may change every round.

Named input distributions (test/bench data generation only):
  TWG_DIST_UNIFORM : U, V bytes approximately uniform int8.
  TWG_DIST_SMALL   : U, V values in [-2, 2].
  TWG_DIST_SPARSE  : ~90% zero bytes.
  TWG_DIST_SATURATE: values mostly in {-127, 127}.
  TWG_DIST_ZERO_U  : U all zero (the salt XOR still makes tasks nonzero).

Inputs (device pointers, fully initialized by the harness):
  U : int8_t [R*32, K] row-major. Row (i*32 + a) feeds tile row a of tile
      row-block i.
  V : int8_t [C*32, K] row-major. Row (j*32 + b) feeds tile column b of tile
      column-block j.

ROUND SEMANTICS (normative, exact):
  Let r = the round counter value BEFORE this run (0 for the first round
  after reset). Let ACC be the state before this round and ACC' after.
  All scalars below are uint32 with wraparound arithmetic; int8/int32
  intermediate products are converted to uint32 by two's-complement
  reinterpretation.

  For each tile (i, j), 0 <= i < R, 0 <= j < C:

    rowvec[b] = (i > 0) ? ACC'[i-1][j][31][b] : 0        for b in [0, 32)
    colvec[a] = (j > 0) ? ACC'[i][j-1][a][31] : 0        for a in [0, 32)
      (note: these read the UPDATED neighbor tiles of THIS round)

    sum_row = sum over b of rowvec[b]                    (uint32 wrap)
    sum_col = sum over a of colvec[a]                    (uint32 wrap)

    s = TWG_SALT_ROW * sum_row
      + TWG_SALT_COL * sum_col
      + TWG_SALT_ROUND * r
      + ((uint32)i << 16) + (uint32)j                    (uint32 wrap)

    sb[m] = byte m of s, little-endian (m = 0..3)

    For a, b in [0, 32):
      G[a][b] = sum over k in [0, K) of
                  int32( (int8_t)( (uint8_t)U[(i*32+a)*K + k] ^ sb[k & 3] ) )
                * int32( (int8_t)V[(j*32+b)*K + k] )
        (exact int32; |G| cannot overflow for K <= 256)

      ACC'[i][j][a][b] = ACC[i][j][a][b]
                       + (uint32)G[a][b]
                       + (uint32)P1 * rowvec[b]
                       + (uint32)P2 * colvec[a]          (uint32 wrap)

  Every tile is updated exactly once per round. The dependency order is
  fully determined by the dataflow above; any execution order consistent
  with it yields identical bytes (the reference uses a GPU-resident
  dependency-counter scheduler).

Outputs after EVERY run:
  round_out[1]   : int32, the round counter AFTER this run (r + 1).
  border_row[x]  : uint32, x in [0, C*32): ACC' global element
                   (row R*32 - 1, column x). Row-major global indexing:
                   global element (y, x) = tile (y/32, x/32), intra (y%32,
                   x%32).
  border_col[y]  : uint32, y in [0, R*32): ACC' global element (y, C*32 - 1).
  state_checksum[1]: uint64, two-level FNV-1a-64 over the FULL ACC' state:
      FNV-1a-64 byte step: h = (h XOR byte) * TWG_FNV_PRIME (mod 2^64),
      seeded with TWG_FNV_BASIS; multi-byte values absorbed little-endian.
      digest[i*C + j] = FNV over tile (i, j) bytes: elements (a, b) in
        a-major order (a ascending, then b ascending), 4 bytes little-endian
        per uint32, seeded with TWG_FNV_BASIS.
      state_checksum = FNV over the bytes of digest[0 .. R*C), each uint64
        absorbed little-endian, ascending tile row-major index, seeded with
        TWG_FNV_BASIS.
  acc_dump       : uint32 [R*32, C*32] row-major (global element layout).
                   Written ONLY when RunSpec.dump == 1; otherwise the buffer
                   must not be touched.

ABI:
  The grader passes TwgRunSpec*, TwgInputs*, TwgOutputs* through the generic
  pipeline ABI. TwgProblemSpec carries maximum allocation bounds.

Rules for solution_run:
  - May launch kernels and use cudaMemsetAsync.
  - May not call cudaMalloc/cudaFree (persistent allocations belong in
    solution_init).
  - May not perform synchronous host/device copies and may not synchronize
    the stream to inspect device values on the host.
  - May not use CUB, Thrust, cuBLAS, cuDNN, or CUTLASS.
  - Must be deterministic: identical call sequences must produce identical
    bytes in every specified output region.
*/

struct alignas(8) TwgProblemSpec {
    int32_t abi_version;
    int32_t max_R;
    int32_t max_C;
    int32_t max_K;
    int32_t flags;
    int32_t reserved[11];
};

struct alignas(8) TwgRunSpec {
    int32_t abi_version;
    int32_t R;
    int32_t C;
    int32_t K;
    int32_t P1;
    int32_t P2;
    int32_t dump;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[6];
};

struct alignas(8) TwgInputs {
    const int8_t* U;
    const int8_t* V;
    const void* reserved0;
    const void* reserved1;
};

struct alignas(8) TwgOutputs {
    int32_t* round_out;      // [1]
    uint32_t* border_row;    // [C * 32]
    uint32_t* border_col;    // [R * 32]
    uint64_t* state_checksum;// [1]
    uint32_t* acc_dump;      // [R*32, C*32], written only when dump == 1
    void* reserved0;
};

static_assert(sizeof(TwgProblemSpec) == 64, "TwgProblemSpec layout drift");
static_assert(sizeof(TwgRunSpec) == 64, "TwgRunSpec layout drift");
static_assert(sizeof(TwgInputs) == 32, "TwgInputs layout drift");
static_assert(sizeof(TwgOutputs) == 48, "TwgOutputs layout drift");

static_assert(offsetof(TwgRunSpec, R) == 4, "layout");
static_assert(offsetof(TwgRunSpec, C) == 8, "layout");
static_assert(offsetof(TwgRunSpec, K) == 12, "layout");
static_assert(offsetof(TwgRunSpec, P1) == 16, "layout");
static_assert(offsetof(TwgRunSpec, P2) == 20, "layout");
static_assert(offsetof(TwgRunSpec, dump) == 24, "layout");

static inline size_t twg_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int twg_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int twg_valid_K(int K) {
    return K == 64 || K == 128 || K == 256;
}

static inline int twg_validate_problem_spec(const TwgProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != TWG_ABI_VERSION) return 0;
    if (spec->max_R < TWG_MIN_R || spec->max_R > TWG_MAX_R) return 0;
    if (spec->max_C < TWG_MIN_C || spec->max_C > TWG_MAX_C) return 0;
    if (!twg_valid_K(spec->max_K)) return 0;
    return 1;
}

static inline int twg_validate_run_spec(const TwgRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != TWG_ABI_VERSION) return 0;
    if (run->R < TWG_MIN_R || run->R > TWG_MAX_R) return 0;
    if (run->C < TWG_MIN_C || run->C > TWG_MAX_C) return 0;
    if (!twg_valid_K(run->K)) return 0;
    if (run->P1 < TWG_MIN_P || run->P1 > TWG_MAX_P) return 0;
    if (run->P2 < TWG_MIN_P || run->P2 > TWG_MAX_P) return 0;
    if (run->dump != 0 && run->dump != 1) return 0;
    if (run->distribution_id < TWG_DIST_UNIFORM ||
        run->distribution_id > TWG_DIST_ZERO_U) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const TwgProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const TwgProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const TwgRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // TASKGRAPH_WAVEFRONT_GEMM_COMMON_H_
