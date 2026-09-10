// file: mxfp4_atom_quant_dot_common.h

#ifndef MXFP4_ATOM_QUANT_DOT_COMMON_H_
#define MXFP4_ATOM_QUANT_DOT_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MXQ_ABI_VERSION 1

#define MXQ_MIN_R 128
#define MXQ_MAX_R 16384
#define MXQ_MIN_C 64
#define MXQ_MAX_C 6144

#define MXQ_BLOCK_K 32
#define MXQ_DOT_LANES 128
#define MXQ_DIGEST_SUBSTREAMS 8

// Project FNV constants. This project intentionally does NOT use the
// canonical FNV-1a-64 offset basis.
#define MXQ_FNV_BASIS 1469598103934665603ull
#define MXQ_FNV_PRIME 1099511628211ull

enum MxqDistributionId : int32_t {
    MXQ_DIST_SMOOTH = 0,
    MXQ_DIST_TIES = 1,
    MXQ_DIST_SATURATE = 2,
    MXQ_DIST_ZERO_BLOCKS = 3,
    MXQ_DIST_DENORMAL = 4,
    MXQ_DIST_NEGZERO = 5,
    MXQ_DIST_POW2_EDGE = 6,
    MXQ_DIST_CLAMP_LOW = 7,
    MXQ_DIST_RANDBITS = 8
};

/*
CONTRACT: mxfp4_atom_quant_dot

Exact MXFP4 (OCP-MX style: FP4 E2M1 payload + per-32-element E8M0 scales)
quantization of an fp32 matrix, packed into CUTLASS-style 128x128 scale-factor
atoms and 128x64 payload atoms, with a fused, deterministically ordered fp32
dequant-dot epilogue, per-row saturation counts, and per-row two-level
FNV-1a-64 digests. Every output is graded exactly (byte-exact / bit-exact).

Inputs (device, read-only):
  x[R, C]  fp32, row-major, contiguous (element (r,k) at x[r*C + k]).
  v[C]     fp32.
All input values are finite or zero EXCEPT under MXQ_DIST_RANDBITS, where any
finite fp32 bit pattern with biased exponent field <= 0xF0 may appear
(normals, subnormals, +-0.0). Infinities and NaNs never appear in x or v.
DOMAIN GUARANTEE: the harness generates x and v so that every product and
every partial-sum intermediate of stage 4 is finite -- overflow to infinity
(and hence NaN) can never occur in a conforming implementation. No other
property of the inputs may be assumed.

Shape family:
  R in [128, 16384], C in [64, 6144]. Neither R nor C is required to be a
  multiple of anything: ragged tails in the 32-element scale blocks, the
  64-element payload atoms, the 128-element scale atoms and the 128-row atom
  tiles all occur.

Derived sizes:
  S  = ceil(C / 32)    scale blocks per row
  Kb = ceil(C / 2)     logical packed payload bytes per row
  AM = ceil(R / 128)   atom row tiles
  AK = ceil(C / 64)    payload atoms per row tile
  AS = ceil(C / 128)   scale atoms per row tile
  pay_atoms buffer size = 4096 * AM * AK bytes
  sf_atoms  buffer size =  512 * AM * AS bytes

--------------------------------------------------------------------------
STAGE 1: per-block E8M0 scale (normative)
--------------------------------------------------------------------------
Scale block b of row r covers k in [32b, min(32b+32, C)), 0 <= b < S.

  amax = max over covered k of fabsf(x[r,k])   (so amax >= +0.0)

If amax == 0.0f (all covered lanes are +-0.0):
  sf_byte(r,b) = 0x00, and the element codes of the block are sign-only
  (magnitude code 0; see stage 2 -- the sign bit of each +-0.0 IS preserved).
  For stage 3/4 the block behaves as if sexp = -127.

Otherwise let e = floor_log2(amax), defined EXACTLY on the fp32 bit pattern
u = bits(amax) (sign bit is 0):
  E = (u >> 23) & 0xff, M = u & 0x7fffff
  if E > 0 : e = E - 127                        (normal)
  else     : e = msb_index(M) - 149             (subnormal; msb_index(1)=0,
                                                 msb_index(0x400000)=22)
Then:
  sexp = e - 2, clamped below to -127 (sexp = max(e - 2, -127)).
  (e <= 127 for finite amax, so sexp <= 125; no upper clamp is reachable.)
  sf_byte(r,b) = sexp + 127   (in [0, 254]; 255 never occurs)

The E8M0 scale value of the block is 2^sexp.

--------------------------------------------------------------------------
STAGE 2: per-element E2M1 code (normative)
--------------------------------------------------------------------------
The E2M1 magnitude lattice, indexed by magnitude code m in [0,7]:
  lat[8] = { 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 }

For element k in block b (sexp from stage 1):
  a = fabsf(x[r,k])
  sign = signbit(x[r,k])        (1 for negative, INCLUDING -0.0)

The magnitude code is round-to-nearest on the lattice lat[m] * 2^sexp with
ties resolved to the EVEN magnitude code, saturating at code 7. Normatively,
with T(t) = t * 2^sexp (each t * 2^sexp below is exactly representable in
fp32 for every reachable sexp in [-127, 125]; comparisons are exact):

  a <  T(0.25) -> m = 0        a == T(0.25) -> m = 0
  a <  T(0.75) -> m = 1        a == T(0.75) -> m = 2
  a <  T(1.25) -> m = 2        a == T(1.25) -> m = 2
  a <  T(1.75) -> m = 3        a == T(1.75) -> m = 4
  a <  T(2.5)  -> m = 4        a == T(2.5)  -> m = 4
  a <  T(3.5)  -> m = 5        a == T(3.5)  -> m = 6
  a <  T(5.0)  -> m = 6        a == T(5.0)  -> m = 6
  otherwise    -> m = 7

  code(r,k) = (sign << 3) | m      (4-bit nibble)

For k >= C (payload/atom padding lanes) code is 0x0 (no sign bit).

Saturation count (exact output):
  sat_count[r] = int32 number of k in [0, C) with
                 fabsf(x[r,k]) > T(6.0)  (strictly greater; T(6.0) = 6*2^sexp
                 of k's block, exactly representable for all reachable sexp).

--------------------------------------------------------------------------
STAGE 3: packed layouts (normative)
--------------------------------------------------------------------------
LOGICAL packed payload stream of row r (used only by stage 5 digests; it is
not itself an output buffer): byte j in [0, Kb):
  low  nibble = code(r, 2j)
  high nibble = code(r, 2j+1) if 2j+1 < C else 0x0
Note: for odd C the final high nibble is ZERO, sign bit included.

PAYLOAD ATOMS (output pay_atoms, byte-exact): 128 rows x 64 elements per
atom, 4096 bytes. Atom (am, ak), 0 <= am < AM, 0 <= ak < AK, starts at
  pay_atom_base = 4096 * (am * AK + ak)
For row r = 128*am + rr (0 <= rr < 128) and element k = 64*ak + j
(0 <= j < 64), with u = j mod 32, one atom byte holds TWO elements:
  j = u      -> low  nibble
  j = u + 32 -> high nibble
(NOTE: the atom nibble pairing (u, u+32) differs from the logical stream
pairing (2j, 2j+1).) The intra-atom byte offset is
  payload_off(rr, u) = (u/8)*1024 + (rr%32)*32 + (rr/32)*8 + (u%8)
Padding nibbles (r >= R or k >= C) are 0x0. EVERY byte of pay_atoms must be
written (all 4096*AM*AK bytes, including fully-padding bytes, which are 0x00).

SCALE ATOMS (output sf_atoms, byte-exact): one scale atom covers 128 rows x
4 scale blocks (= 128 elements), 512 bytes. Atom (am, as), 0 <= as < AS,
starts at
  sf_atom_base = 512 * (am * AS + as)
For row r = 128*am + rr and scale block s = 4*as + c (0 <= c < 4):
  scale_off(rr, c) = (rr%32)*16 + (rr/32)*4 + c
holds sf_byte(r, s). Padding bytes (r >= R or s >= S) are 0x00. EVERY byte
of sf_atoms must be written.

--------------------------------------------------------------------------
STAGE 4: fused deterministic dequant-dot (normative, bit-exact fp32)
--------------------------------------------------------------------------
Dequantized value of element (r,k), exact in fp32:
  dq(r,k) = (sign(r,k) ? -1 : +1) * lat[m(r,k)] * 2^sexp(block of k)
  (lat[m] * 2^sexp is exactly representable for all reachable sexp; for
  m = 0 this is +-0.0 with the sign of x[r,k].)

row_dot[r] is computed with MXQ_DOT_LANES = 128 pinned partial lanes:
  partial[l] (l in [0,128)) starts at +0.0f and accumulates over
  k = l, l+128, l+256, ... (< C), in increasing k:
      partial[l] = fadd_rn(partial[l], fmul_rn(dq(r,k), v[k]))
  where fadd_rn/fmul_rn are IEEE-754 fp32 add/multiply, round-to-nearest-
  even, NO FMA contraction, subnormals preserved (no FTZ).
Then a binary tree, in this exact order:
  for stride = 64, 32, 16, 8, 4, 2, 1:
      for l in [0, stride): partial[l] = fadd_rn(partial[l], partial[l+stride])
  row_dot[r] = partial[0]
row_dot is graded BIT-exact (as u32 bit patterns).

--------------------------------------------------------------------------
STAGE 5: per-row two-level FNV-1a-64 digest (normative, exact)
--------------------------------------------------------------------------
fold(h, byte) = (h XOR byte) * MXQ_FNV_PRIME   (mod 2^64)
Multi-byte words are folded little-endian, full width (low byte first).

The digest stream of row r is the Kb logical payload bytes (stage 3, j
ascending) followed by the S scale bytes sf_byte(r, 0..S-1):
  total_len = Kb + S, stream[i] = (i < Kb) ? pay_logical(r,i) : sf_byte(r,i-Kb)

The stream is split into MXQ_DIGEST_SUBSTREAMS = 8 ranges:
  sub_len = ceil(total_len / 8)
  range g = [g*sub_len, min((g+1)*sub_len, total_len)), g in [0, 8)
  (empty ranges allowed; their sub-digest is the basis.)
  sub[g] = fold of MXQ_FNV_BASIS over the bytes of range g, in order.
Then:
  h = MXQ_FNV_BASIS
  for g = 0..7: fold the 8 bytes of sub[g], little-endian (low byte first)
  row_digest[r] = h                (graded exactly as u64)

--------------------------------------------------------------------------
Outputs (all graded, no tolerances anywhere):
  pay_atoms  uint8 [4096*AM*AK]   byte-exact (padding = 0x00)
  sf_atoms   uint8 [ 512*AM*AS]   byte-exact (padding = 0x00)
  row_dot    fp32  [R]            bit-exact (u32 pattern)
  row_digest uint64[R]            exact
  sat_count  int32 [R]            exact

Rules:
  - solution_init may allocate persistent state sized from MxqProblemSpec.
  - solution_run must NOT call cudaMalloc/cudaFree/cudaMallocAsync or any
    host<->device memcpy (cudaMemsetAsync on device buffers is allowed);
    it launches kernels and may use the provided workspace.
  - Inputs are read-only and must not be modified.
  - No CUB, Thrust, cuBLAS, cuBLASLt, cuDNN or CUTLASS (headers included);
    plain CUDA C++ only.
  - The reference is a fused pipeline: it never materializes the fp32
    dequantized matrix or per-element codes in global memory.
*/

struct alignas(8) MxqProblemSpec {
    int32_t abi_version;
    int32_t max_R;
    int32_t max_C;
    int32_t flags;
    int32_t reserved[12];
};

struct alignas(8) MxqRunSpec {
    int32_t abi_version;
    int32_t R;
    int32_t C;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[10];
};

struct alignas(8) MxqInputs {
    const float* x;
    const float* v;
};

struct alignas(8) MxqOutputs {
    uint8_t* pay_atoms;
    uint8_t* sf_atoms;
    float* row_dot;
    uint64_t* row_digest;
    int32_t* sat_count;
};

static_assert(sizeof(MxqProblemSpec) == 64, "MxqProblemSpec layout drift");
static_assert(sizeof(MxqRunSpec) == 64, "MxqRunSpec layout drift");
static_assert(sizeof(MxqInputs) == 16, "MxqInputs layout drift");
static_assert(sizeof(MxqOutputs) == 40, "MxqOutputs layout drift");

static inline size_t mxq_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int mxq_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int mxq_S(int C) { return (C + 31) / 32; }
static inline int mxq_Kb(int C) { return (C + 1) / 2; }
static inline int mxq_AM(int R) { return (R + 127) / 128; }
static inline int mxq_AK(int C) { return (C + 63) / 64; }
static inline int mxq_AS(int C) { return (C + 127) / 128; }

static inline size_t mxq_pay_atom_bytes(int R, int C) {
    return (size_t)4096 * (size_t)mxq_AM(R) * (size_t)mxq_AK(C);
}

static inline size_t mxq_sf_atom_bytes(int R, int C) {
    return (size_t)512 * (size_t)mxq_AM(R) * (size_t)mxq_AS(C);
}

static inline int mxq_validate_problem_spec(const MxqProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MXQ_ABI_VERSION) return 0;
    if (spec->max_R < MXQ_MIN_R || spec->max_R > MXQ_MAX_R) return 0;
    if (spec->max_C < MXQ_MIN_C || spec->max_C > MXQ_MAX_C) return 0;
    return 1;
}

static inline int mxq_validate_run_spec(const MxqRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != MXQ_ABI_VERSION) return 0;
    if (run->R < MXQ_MIN_R || run->R > MXQ_MAX_R) return 0;
    if (run->C < MXQ_MIN_C || run->C > MXQ_MAX_C) return 0;
    if (run->distribution_id < MXQ_DIST_SMOOTH ||
        run->distribution_id > MXQ_DIST_RANDBITS) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const MxqProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const MxqProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const MxqRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // MXFP4_ATOM_QUANT_DOT_COMMON_H_
