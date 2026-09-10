// file: sat_segscan_select_common.h

#ifndef SAT_SEGSCAN_SELECT_COMMON_H_
#define SAT_SEGSCAN_SELECT_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define SSS_ABI_VERSION 1

#define SSS_MIN_N 65536
#define SSS_MAX_N 33554432   // 2^25

#define SSS_MIN_LO (-16777216)  // -2^24
#define SSS_MAX_LO (-1)
#define SSS_MIN_HI (1)
#define SSS_MAX_HI (16777216)   // 2^24

#define SSS_MAX_ABS_V (1048576)  // 2^20

enum SssDistributionId : int32_t {
    SSS_DIST_SMOOTH = 0,
    SSS_DIST_TIGHT = 1,
    SSS_DIST_SATRUN = 2,
    SSS_DIST_EXACT = 3,
    SSS_DIST_ONESEG = 4,
    SSS_DIST_ALLSEG = 5,
    SSS_DIST_RAGGED = 6,
    SSS_DIST_ZERO = 7,
    SSS_DIST_RANDBITS = 8
};

/*
CONTRACT: sat_segscan_select

Exact SEGMENTED SATURATING inclusive scan over an int32 stream with a packed
head-flag bit stream, producing (1) the dense scan values, (2) a packed
saturation bitmap, (3) the ordered compaction of all saturated positions,
and (4) the last scan value of every segment. Every graded output is exact
(bit/byte/integer) -- there are no tolerances anywhere.

Dimensions:
  N elements, N in [65536, 33554432].
  Nw = ceil(N / 32) packed flag words.
  S  = number of set flag bits (segment count; see below).

Inputs (device, read-only):
  v     int32 [N]    element values, |v[i]| <= 2^20.
  flags uint32[Nw]   packed head flags, little-endian bit order:
                     flag(i) = (flags[i >> 5] >> (i & 31)) & 1.
                     GUARANTEES: flag(0) == 1 (the stream starts with a
                     segment head) and all padding bits at positions >= N in
                     the last word are 0. Any other bit pattern may appear:
                     runs of zeros spanning millions of elements, all-ones,
                     and everything between.

Run parameters (SssRunSpec): saturation rails LO and HI with
  LO in [-2^24, -1],  HI in [1, 2^24].

--------------------------------------------------------------------------
NORMATIVE SEMANTICS (sequential; defines every output exactly)
--------------------------------------------------------------------------
clamp(t) = min(max(t, LO), HI), computed exactly (with the input bounds
above no intermediate exceeds int32 range: |p| <= 2^24 and |v| <= 2^20).

  p = 0
  ord = -1                       // current segment ordinal
  for i = 0 .. N-1:
      if flag(i): { p = 0; ord = ord + 1 }
      p = clamp(p + v[i])
      y[i] = p
      sat(i) = (p == LO || p == HI)      // value equality, NOT "was clamped"
      if i == N-1 or flag(i+1): seg_last[ord] = p

  sel stream: the global indices i with sat(i) == 1, in ascending order.
  sel_count = number of such indices.
  S = ord + 1 = popcount of all flag bits.

--------------------------------------------------------------------------
Outputs (graded exactly):
  y         int32 [N]    dense scan values (exact).
  sat_bits  uint32[Nw]   packed sat(i) bits, same bit order as flags;
                         padding bits at positions >= N MUST be 0.
                         Graded byte-exact over the whole buffer.
  sel_idx   int32 [N]    sel_idx[j] = j-th saturated index, for
                         j in [0, sel_count). Only the first sel_count
                         entries are graded; entries beyond sel_count are
                         never read (the capacity is always N).
  sel_count int32 [1]    exact.
  seg_last  int32 [S]    exact (the harness sizes the buffer to exactly S).

Rules:
  - solution_init may allocate persistent state sized from SssProblemSpec.
  - solution_run must NOT call cudaMalloc/cudaFree/cudaMallocAsync or any
    host<->device memcpy (cudaMemsetAsync on device buffers is allowed);
    it launches kernels and may use the provided workspace.
  - Inputs are read-only and must not be modified.
  - No CUB, Thrust, cuBLAS, cuBLASLt, cuDNN or CUTLASS (headers included);
    plain CUDA C++ only. Any scan/decoupled-lookback machinery must be your
    own device code.
  - Determinism: the same inputs must always produce the same outputs.

NOTE (what makes this hard): saturating addition is NOT associative --
((a satadd b) satadd c) != (a satadd (b satadd c)) in general -- so the
textbook aggregate-passing scan over VALUES is simply wrong. The sequential
semantics above are the single source of truth; any parallel decomposition
must reproduce them bit-for-bit on segment runs that span millions of
elements (a single segment covering the whole stream occurs). The task is
perf-gated against a single-pass reference, so per-segment sequential
fallbacks do not survive the gate.
*/

struct alignas(8) SssProblemSpec {
    int32_t abi_version;
    int32_t max_N;
    int32_t flags;
    int32_t reserved[13];
};

struct alignas(8) SssRunSpec {
    int32_t abi_version;
    int32_t N;
    int32_t lo;
    int32_t hi;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[9];
};

struct alignas(8) SssInputs {
    const int32_t* v;
    const uint32_t* flags;
};

struct alignas(8) SssOutputs {
    int32_t* y;
    uint32_t* sat_bits;
    int32_t* sel_idx;
    int32_t* sel_count;
    int32_t* seg_last;
};

static_assert(sizeof(SssProblemSpec) == 64, "SssProblemSpec layout drift");
static_assert(sizeof(SssRunSpec) == 64, "SssRunSpec layout drift");
static_assert(sizeof(SssInputs) == 16, "SssInputs layout drift");
static_assert(sizeof(SssOutputs) == 40, "SssOutputs layout drift");

static inline int sss_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int sss_Nw(int N) { return (N + 31) / 32; }

static inline int sss_validate_problem_spec(const SssProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != SSS_ABI_VERSION) return 0;
    if (spec->max_N < SSS_MIN_N || spec->max_N > SSS_MAX_N) return 0;
    return 1;
}

static inline int sss_validate_run_spec(const SssRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != SSS_ABI_VERSION) return 0;
    if (run->N < SSS_MIN_N || run->N > SSS_MAX_N) return 0;
    if (run->lo < SSS_MIN_LO || run->lo > SSS_MAX_LO) return 0;
    if (run->hi < SSS_MIN_HI || run->hi > SSS_MAX_HI) return 0;
    if (run->distribution_id < SSS_DIST_SMOOTH ||
        run->distribution_id > SSS_DIST_RANDBITS) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const SssProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const SssProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const SssRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // SAT_SEGSCAN_SELECT_COMMON_H_
