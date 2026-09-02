// file: microscale_requant_chain_common.h

#ifndef MICROSCALE_REQUANT_CHAIN_COMMON_H_
#define MICROSCALE_REQUANT_CHAIN_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MRQ_ABI_VERSION 1

#define MRQ_MIN_R 128
#define MRQ_MAX_R 8192
#define MRQ_MIN_C 64
#define MRQ_MAX_C 4096

#define MRQ_BLOCK_K 32
#define MRQ_DIGEST_SUBSTREAMS 8

// Project FNV constants. This project intentionally does NOT use the
// canonical FNV-1a-64 offset basis.
#define MRQ_FNV_BASIS 1469598103934665603ull
#define MRQ_FNV_PRIME 1099511628211ull

enum MrqDistributionId : int32_t {
    MRQ_DIST_SMOOTH = 0,
    MRQ_DIST_TIES1 = 1,
    MRQ_DIST_SAT = 2,
    MRQ_DIST_TIES2 = 3,
    MRQ_DIST_ZEROBLK = 4,
    MRQ_DIST_DENORM = 5,
    MRQ_DIST_NEGZ = 6,
    MRQ_DIST_POW2 = 7,
    MRQ_DIST_RANDBITS = 8
};

/*
CONTRACT: microscale_requant_chain

A deterministic two-stage microscaling requantization chain:
  STAGE 1: fp32 -> per-32-element E8M0 scale + FP8 E4M3 element codes
           (round-to-nearest-even at the bit level, saturating at 448,
           subnormals and signed zeros preserved).
  STAGE 2: the EXACT stage-1 output values -> per-32-element power-of-two
           second scale + signed int4 codes through a non-uniform 8-level
           lookup table, rounding to the NEAREST level with ties to the
           LARGER index (a different tie rule than stage 1).
Plus per-row exact integer squared requant error, per-row stage-1
saturation counts, and per-row two-level word-folded FNV-1a-64 digests.
Every output is graded exactly (byte-exact / exact integers) -- there are
no tolerances anywhere. Every rounding decision below is exactly decidable
in integer arithmetic; a bit-level integer implementation is recommended.

Inputs (device, read-only):
  x[R, C]  fp32, row-major, contiguous (element (r,k) at x[r*C + k]).
All values are finite (biased exponent field <= 0xFE; infinities and NaNs
never appear). Subnormals and +-0.0 DO appear. No other property of the
inputs may be assumed.

Shape family:
  R in [128, 8192], C in [64, 4096]. Neither is required to be a multiple
  of anything: ragged tails in the 32-element scale blocks and the 2-per-
  byte int4 packing occur.

Derived sizes:
  S  = ceil(C / 32)    scale blocks per row
  Kb = ceil(C / 2)     packed int4 bytes per row

--------------------------------------------------------------------------
E4M3 (normative; OCP FP8 E4M3 without NaN emission)
--------------------------------------------------------------------------
code byte = (sign << 7) | (E << 3) | m,  E in [0,15], m in [0,7].
  E >= 1: value = (8 + m) * 2^(E - 10)      (i.e. 1.m * 2^(E-7))
  E == 0: value = m * 2^(-9)                (subnormal; m=0 is +-0.0)
Max magnitude 448 (E=15, m=6). The code (E=15, m=7) is NaN in OCP FP8 and
is NEVER produced by this task. Every E4M3 magnitude is an integer
multiple of 2^-9; define the integer magnitude
  M(code) = value * 2^9   (an integer in [0, 229376]).

--------------------------------------------------------------------------
STAGE 1: per-block E8M0 scale + E4M3 codes (normative)
--------------------------------------------------------------------------
Scale block b of row r covers k in [32b, min(32b+32, C)), 0 <= b < S.

  amax = max over covered k of fabsf(x[r,k])   (so amax >= +0.0)

If amax == 0.0f (all covered lanes are +-0.0):
  sf1(r,b) = 0x00 and every element code of the block is sign-only
  (E=0, m=0; the sign bit of -0.0 IS preserved). The block behaves as if
  sexp1 = -127 for everything below.

Otherwise let e = floor_log2(amax), defined EXACTLY on the fp32 bit
pattern u = bits(amax):
  E = (u >> 23) & 0xff, Mn = u & 0x7fffff
  if E > 0 : e = E - 127                     (normal)
  else     : e = msb_index(Mn) - 149         (subnormal; msb_index(1)=0)
Then:
  sexp1 = max(e - 8, -127)
  sf1(r,b) = sexp1 + 127     (in [0, 254])
The stage-1 scale value is 2^sexp1.

Element quantization, for k in block b:
  u = |x[r,k]| * 2^-sexp1    (an exact mathematical value; every decision
      below is exactly decidable from the bits of x and sexp1)
  q = RNE_e4m3(u): round u to the nearest E4M3 magnitude on the grid
      above; exact midpoints round to the candidate with EVEN integer
      significand (standard round-to-nearest-even on the grid, including
      across binade boundaries and at the subnormal/normal boundary;
      u == 2^-10 exactly rounds to 0). SATURATION: if the rounded
      magnitude would exceed 448 -- equivalently u > 448 rounds beyond
      448 -- the code is clamped to (E=15, m=6), value 448. NaN is never
      produced.
  e4m3(r,k) = (signbit(x[r,k]) << 7) | code(q)   (sign of -0.0 preserved)

sat1_count[r] = int32 number of k in [0, C) with u > 448 (STRICTLY
  greater, in the element's own block; u == 448 exactly is NOT counted).

Because sexp1 = e - 8 (when unclamped), the block maximum has
u in [256, 512), so the saturation zone (448, 512) is reachable and both
tails of every binade occur.

--------------------------------------------------------------------------
STAGE 2: per-block int4 LUT requant of the E4M3 values (normative)
--------------------------------------------------------------------------
Stage 2 operates on the integer magnitudes M(r,k) = M(e4m3(r,k)) of the
stage-1 OUTPUT (not on x).

Per block b:  qmax = max over covered k of M(r,k).
If qmax == 0: sf2(r,b) = 0xFF and every int4 code of the block is
  sign-only (idx = 0, sign preserved from the e4m3 code). This includes
  blocks whose inputs were nonzero but quantized to zero.
Otherwise:
  msb = msb_index(qmax)      (in [0, 17]; msb_index(1) = 0)
  sf2(r,b) = msb             (equivalently sexp2 + 9 with
                              sexp2 = floor_log2(qmax * 2^-9))

LUT levels (magnitudes, in units of 2^sexp2 = 2^(msb-9)):
  L[8] = { 0, 1/8, 1/4, 3/8, 5/8, 1, 3/2, 2 }
Decision boundaries are the exact midpoints
  B[7] = { 1/16, 3/16, 5/16, 1/2, 13/16, 5/4, 7/4 }.
With w = M(r,k) * 2^-(msb) * ... precisely: w = M(r,k) / 2^msb is the
element magnitude in units of 2^sexp2. The index is nearest-level with
ties to the LARGER index; normatively (all comparisons exact integers,
e.g. 16*M >= p * 2^msb with p in {1,3,5,8,13,20,28}):

  idx = number of boundaries B[j] with w >= B[j]     (j in [0,7))

  q4(r,k) = (sign(e4m3(r,k)) << 3) | idx   (4-bit code)
For k >= C (padding), q4 nibble is 0x0 (no sign bit).

PACKED int4 stream (output q4_packed, byte-exact): byte j in [0, Kb):
  low  nibble = q4(r, 2j)
  high nibble = q4(r, 2j+1) if 2j+1 < C else 0x0
(For odd C the final high nibble is ZERO, sign bit included.)

--------------------------------------------------------------------------
STAGE 3: exact integer squared requant error (normative)
--------------------------------------------------------------------------
Fixed point at 2^-12 in the stage-1 scaled domain (per element, in its
block; zero blocks contribute 0):
  a = M(r,k) * 8                          (= e4m3 magnitude * 2^12)
  rterm = p(idx) << msb, with p(idx) in { 0, 1, 2, 3, 5, 8, 12, 16 }
          (= L[idx] * 8; so rterm = L[idx] * 2^sexp2 * 2^12)
  err = a - rterm                          (signed integer)
  row_err[r] = int64 sum over k in [0, C) of err * err   (order-free,
      exact; bounded well inside int64)

--------------------------------------------------------------------------
STAGE 4: per-row two-level word-folded FNV-1a-64 digest (normative)
--------------------------------------------------------------------------
fold(h, w) = (h XOR w) * MRQ_FNV_PRIME  (mod 2^64), w a full 64-bit word.

The BYTE stream of row r has Lb = C + Kb + 2*S bytes:
  [ C   bytes of e4m3(r, 0..C-1) ]
  [ Kb  bytes of q4_packed row r ]
  [ S   bytes of sf1(r, 0..S-1)  ]
  [ S   bytes of sf2(r, 0..S-1)  ]
Packed little-endian into W = ceil(Lb / 8) 64-bit words (bytes past Lb
are 0). Word w belongs to substream w mod 8:
  sub[t] = fold of MRQ_FNV_BASIS over words w = t, t+8, ... (< W),
           ascending.
  h = MRQ_FNV_BASIS; for t = 0..7: h = fold(h, sub[t]);
  row_digest[r] = h   (graded exactly as u64)

--------------------------------------------------------------------------
Outputs (all graded, no tolerances anywhere):
  e4m3_codes uint8 [R*C]    byte-exact
  q4_packed  uint8 [R*Kb]   byte-exact (odd-C high nibble 0)
  sf1        uint8 [R*S]    byte-exact
  sf2        uint8 [R*S]    byte-exact (0xFF for all-zero-code blocks)
  row_err    int64 [R]      exact
  row_digest uint64[R]      exact
  sat1_count int32 [R]      exact

Rules:
  - solution_init may allocate persistent state sized from MrqProblemSpec.
  - solution_run must NOT call cudaMalloc/cudaFree/cudaMallocAsync or any
    host<->device memcpy (cudaMemsetAsync on device buffers is allowed);
    it launches kernels and may use the provided workspace.
  - Inputs are read-only and must not be modified.
  - No CUB, Thrust, cuBLAS, cuBLASLt, cuDNN or CUTLASS (headers included);
    plain CUDA C++ only.
  - The reference is a fused pipeline: it reads x exactly once and never
    materializes intermediate element matrices in global memory.
*/

struct alignas(8) MrqProblemSpec {
    int32_t abi_version;
    int32_t max_R;
    int32_t max_C;
    int32_t flags;
    int32_t reserved[12];
};

struct alignas(8) MrqRunSpec {
    int32_t abi_version;
    int32_t R;
    int32_t C;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[10];
};

struct alignas(8) MrqInputs {
    const float* x;
};

struct alignas(8) MrqOutputs {
    uint8_t* e4m3_codes;
    uint8_t* q4_packed;
    uint8_t* sf1;
    uint8_t* sf2;
    int64_t* row_err;
    uint64_t* row_digest;
    int32_t* sat1_count;
};

static_assert(sizeof(MrqProblemSpec) == 64, "MrqProblemSpec layout drift");
static_assert(sizeof(MrqRunSpec) == 64, "MrqRunSpec layout drift");
static_assert(sizeof(MrqInputs) == 8, "MrqInputs layout drift");
static_assert(sizeof(MrqOutputs) == 56, "MrqOutputs layout drift");

static inline int mrq_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int mrq_S(int C) { return (C + 31) / 32; }
static inline int mrq_Kb(int C) { return (C + 1) / 2; }
static inline int mrq_stream_bytes(int C) {
    return C + mrq_Kb(C) + 2 * mrq_S(C);
}

static inline int mrq_validate_problem_spec(const MrqProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MRQ_ABI_VERSION) return 0;
    if (spec->max_R < MRQ_MIN_R || spec->max_R > MRQ_MAX_R) return 0;
    if (spec->max_C < MRQ_MIN_C || spec->max_C > MRQ_MAX_C) return 0;
    return 1;
}

static inline int mrq_validate_run_spec(const MrqRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != MRQ_ABI_VERSION) return 0;
    if (run->R < MRQ_MIN_R || run->R > MRQ_MAX_R) return 0;
    if (run->C < MRQ_MIN_C || run->C > MRQ_MAX_C) return 0;
    if (run->distribution_id < MRQ_DIST_SMOOTH ||
        run->distribution_id > MRQ_DIST_RANDBITS) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const MrqProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const MrqProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const MrqRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // MICROSCALE_REQUANT_CHAIN_COMMON_H_
