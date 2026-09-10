// file: awq_actorder_repack_gemv_common.h

#ifndef AWQ_ACTORDER_REPACK_GEMV_COMMON_H_
#define AWQ_ACTORDER_REPACK_GEMV_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define AWQ_ABI_VERSION 1

#define AWQ_MIN_K 256
#define AWQ_MAX_K 8192
#define AWQ_MIN_N 64
#define AWQ_MAX_N 4096
#define AWQ_MIN_G 4
#define AWQ_MAX_G 512

#define AWQ_DOT_LANES 16
#define AWQ_DIGEST_SUBSTREAMS 8

// Project FNV constants. This project intentionally does NOT use the
// canonical FNV-1a-64 offset basis.
#define AWQ_FNV_BASIS 1469598103934665603ull
#define AWQ_FNV_PRIME 1099511628211ull

enum AwqDistributionId : int32_t {
    AWQ_DIST_BALANCED = 0,
    AWQ_DIST_REVERSED = 1,
    AWQ_DIST_SKEWED = 2,
    AWQ_DIST_ALLSAME = 3,
    AWQ_DIST_ZEROMATCH = 4,
    AWQ_DIST_ZEROSCALE = 5,
    AWQ_DIST_DENORM = 6,
    AWQ_DIST_POW2 = 7,
    AWQ_DIST_RANDBITS = 8
};

/*
CONTRACT: awq_actorder_repack_gemv

AWQ-style int4 weights (packed along the output-channel dimension) with
grouped int4 zero points and fp32 scales are repacked, GPTQ act-order style,
into marlin-flavoured 16x64 tiles along a STABLE group-sorted permutation of
the input channels, fused with an exact per-output-column dequantized dot
product against an activation vector, exact per-column integer sums, and
per-column two-level FNV-1a-64 digests over the permuted dequant-integer
stream. Every output is graded exactly (byte-exact / bit-exact / exact
integers) -- there are no tolerances anywhere.

Dimensions:
  K input channels,  K in [256, 8192]
  N output channels, N in [64, 4096]
  G quantization groups, G in [4, 512] and G <= K

Derived sizes:
  Nw = ceil(N / 8)     packed source words per row
  TA = ceil(K / 16)    permuted-row tiles
  TN = ceil(N / 64)    output-column tiles
  Kp = 16 * TA         padded permuted length (internal definition only)
  rq_atoms buffer size = 512 * TA * TN bytes (= 128 * TA * TN uint32 words)

Inputs (device, read-only):
  qweight uint32[K * Nw]   row-major: word (k, c) at qweight[k*Nw + c].
      Word (k, c) stores the eight unsigned int4 weights q(k, n) of output
      channels n = 8c + t, t in [0, 8). The logical lane t sits at physical
      nibble lane wlane(t) = ((t & 3) << 1) | (t >> 2), i.e. logical lanes
      0..7 occupy physical nibble lanes 0,2,4,6,1,3,5,7 (AWQ weight order):
          q(k, 8c + t) = (qweight[k*Nw + c] >> (4 * wlane(t))) & 0xF
      Padding lanes (8c + t >= N) are guaranteed to hold nibble value 0.
  qzeros  uint32[G * Nw]   row-major: word (g, c) at qzeros[g*Nw + c].
      Same 8-channel packing but with the AWQ zero order
      zlane(t) = ((t & 1) << 2) | (t >> 1), i.e. logical lanes 0..7 occupy
      physical nibble lanes 0,4,1,5,2,6,3,7:
          z(g, 8c + t) = (qzeros[g*Nw + c] >> (4 * zlane(t))) & 0xF
      Padding lanes (8c + t >= N) are guaranteed to hold nibble value 0.
  scales  float[G * N]     row-major: s(g, n) = scales[g*N + n].
  g_idx   int32[K]         act-order group map: g_idx[k] in [0, G).
      NOT sorted; arbitrary order, duplicates and empty groups occur.
  x       float[K]         activation vector.

All scales and x values are finite (never inf/NaN) with biased exponent
field <= 0x9F (|value| < 2^33); +-0.0 and subnormals DO occur, and scales
may be negative or exactly +-0.0. q and z nibbles are arbitrary in [0, 15].
DOMAIN GUARANTEE: with these bounds, every product and every partial-sum
intermediate of stage 3 is finite -- overflow to infinity (and hence NaN)
can never occur in a conforming implementation. No other property of the
inputs may be assumed.

--------------------------------------------------------------------------
STAGE 1: act-order permutation (normative; graded output)
--------------------------------------------------------------------------
perm is the stable ascending argsort of g_idx: channels are ordered by
group index, ties broken by original channel index. Normatively, for each
k in [0, K):

  rank(k) = |{ k' in [0,K) : g_idx[k'] <  g_idx[k] }|
          + |{ k' <  k     : g_idx[k'] == g_idx[k] }|

  perm[rank(k)] = k          (ranks are a bijection on [0, K))

perm (int32[K]) is itself a graded output, and defines the permuted
position j: position j holds source channel perm[j]. Everything below
that says "j" iterates permuted positions.

Shorthand used below (NOT buffers; per (j, n) values):
  gq(j)     = g_idx[perm[j]]
  q(j, n)   = weight nibble of (perm[j], n)          (stage-0 unpack)
  z(j, n)   = z(gq(j), n)                            (stage-0 unpack)
  qz(j, n)  = (int) q(j, n) - (int) z(j, n)          (in [-15, 15])

--------------------------------------------------------------------------
STAGE 2: marlin-style tile repack (normative; byte-exact output)
--------------------------------------------------------------------------
rq_atoms is a uint32 word array (little-endian byte view is graded).
Atom (ta, tn), 0 <= ta < TA, 0 <= tn < TN, covers permuted positions
j in [16*ta, 16*ta + 16) and output channels n in [64*tn, 64*tn + 64),
and owns the 128 words starting at

  atom_word_base(ta, tn) = 128 * (ta * TN + tn)

Within the atom, for h in {0, 1} and nt in [0, 64), the word at offset

  word_off(h, nt) = h * 64 + (((nt & 7) << 3) | (nt >> 3))

packs the eight weights q(j, n) with j = 16*ta + 8*h + i (i in [0, 8)) and
n = 64*tn + nt, placing element i at physical nibble lane
wlane(i) = ((i & 3) << 1) | (i >> 2):

  word = sum over i of  q(16*ta + 8*h + i, 64*tn + nt) << (4 * wlane(i))

Out-of-range elements (j >= K or n >= N) contribute nibble 0. EVERY word
of rq_atoms must be written (all 128 * TA * TN words, including words that
are entirely padding, which are 0x00000000).

NOTE the three distinct layouts in play: the SOURCE packs 8 output channels
per word (wlane along n), the DESTINATION packs 8 permuted input channels
per word (wlane along j) with the h/i row split and the (nt & 7)/(nt >> 3)
column swizzle, and the zero points use zlane. Do not mix them up.

--------------------------------------------------------------------------
STAGE 3: exact dequantized dot (normative; bit-exact fp32) + integer sums
--------------------------------------------------------------------------
For each output column n, col_dot[n] is computed with AWQ_DOT_LANES = 16
pinned partial lanes over PERMUTED positions:
  partial[l] (l in [0,16)) starts at +0.0f and accumulates over
  j = l, l+16, l+32, ... (< K), in increasing j:
      w = fmul_rn( (float) qz(j, n), s(gq(j), n) )
      p = fmul_rn( w, x[perm[j]] )
      partial[l] = fadd_rn( partial[l], p )
  where (float) qz is the exact int-to-fp32 conversion and fmul_rn/fadd_rn
  are IEEE-754 fp32 multiply/add, round-to-nearest-even, NO FMA contraction,
  subnormals preserved (no FTZ). Signed zeros follow IEEE-754 exactly.
Then a binary tree, in this exact order:
  for stride = 8, 4, 2, 1:
      for l in [0, stride): partial[l] = fadd_rn(partial[l], partial[l+stride])
  col_dot[n] = partial[0]
col_dot is graded BIT-exact (as u32 bit patterns).

col_zsum[n] = int32 sum over all j in [0, K) of qz(j, n)   (order-free,
exact; |col_zsum[n]| <= 15*K fits int32 trivially).

--------------------------------------------------------------------------
STAGE 4: per-column two-level word-folded FNV-1a-64 digest (normative)
--------------------------------------------------------------------------
fold(h, w) = (h XOR w) * AWQ_FNV_PRIME   (mod 2^64), w a full 64-bit word.

The BYTE stream of column n has L = K + 4*G bytes:
  stream[i] = (uint8_t)(int8_t) qz(i, n)            for i in [0, K)
              (two's-complement low byte of qz at permuted position j = i)
  stream[K + 4*g + b] = byte b (little-endian) of the fp32 bit pattern of
              scales[g*N + n],  g in [0, G), b in [0, 4)
              (ALL G groups appear, including empty ones.)

The byte stream is packed little-endian into the WORD stream of
W = ceil(L / 8) 64-bit words:
  word[i] = sum over b in [0, 8) of  stream[8i + b] << (8*b)
            (bytes at positions >= L are 0; when K % 8 != 0 one word
            straddles the qz/scale boundary and mixes both sources.)

The word stream is distributed round-robin over
AWQ_DIGEST_SUBSTREAMS = 8 substreams: word w belongs to substream w mod 8.
  sub[r] = fold of AWQ_FNV_BASIS over words w = r, r+8, r+16, ... (< W),
           in ascending w.
Then:
  h = AWQ_FNV_BASIS
  for r = 0..7: h = fold(h, sub[r])
  col_digest[n] = h                (graded exactly as u64)

--------------------------------------------------------------------------
Outputs (all graded, no tolerances anywhere):
  rq_atoms   uint8 [512*TA*TN]   byte-exact (padding words = 0x00000000)
  col_dot    fp32  [N]           bit-exact (u32 pattern)
  col_digest uint64[N]           exact
  col_zsum   int32 [N]           exact
  perm       int32 [K]           exact

Rules:
  - solution_init may allocate persistent state sized from AwqProblemSpec.
  - solution_run must NOT call cudaMalloc/cudaFree/cudaMallocAsync or any
    host<->device memcpy (cudaMemsetAsync on device buffers is allowed);
    it launches kernels and may use the provided workspace.
  - Inputs are read-only and must not be modified.
  - No CUB, Thrust, cuBLAS, cuBLASLt, cuDNN or CUTLASS (headers included);
    plain CUDA C++ only.
  - The reference never materializes the unpacked q matrix in global
    memory and reads qweight exactly once per call.
*/

struct alignas(8) AwqProblemSpec {
    int32_t abi_version;
    int32_t max_K;
    int32_t max_N;
    int32_t max_G;
    int32_t flags;
    int32_t reserved[11];
};

struct alignas(8) AwqRunSpec {
    int32_t abi_version;
    int32_t K;
    int32_t N;
    int32_t G;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[9];
};

struct alignas(8) AwqInputs {
    const uint32_t* qweight;
    const uint32_t* qzeros;
    const float* scales;
    const int32_t* g_idx;
    const float* x;
};

struct alignas(8) AwqOutputs {
    uint8_t* rq_atoms;
    float* col_dot;
    uint64_t* col_digest;
    int32_t* col_zsum;
    int32_t* perm;
};

static_assert(sizeof(AwqProblemSpec) == 64, "AwqProblemSpec layout drift");
static_assert(sizeof(AwqRunSpec) == 64, "AwqRunSpec layout drift");
static_assert(sizeof(AwqInputs) == 40, "AwqInputs layout drift");
static_assert(sizeof(AwqOutputs) == 40, "AwqOutputs layout drift");

static inline int awq_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int awq_Nw(int N) { return (N + 7) / 8; }
static inline int awq_TA(int K) { return (K + 15) / 16; }
static inline int awq_TN(int N) { return (N + 63) / 64; }

static inline size_t awq_rq_bytes(int K, int N) {
    return (size_t)512 * (size_t)awq_TA(K) * (size_t)awq_TN(N);
}

// AWQ weight order: logical lanes 0..7 -> physical nibbles 0,2,4,6,1,3,5,7.
static inline int awq_wlane(int t) { return ((t & 3) << 1) | (t >> 2); }

// AWQ zero order: logical lanes 0..7 -> physical nibbles 0,4,1,5,2,6,3,7.
static inline int awq_zlane(int t) { return ((t & 1) << 2) | (t >> 1); }

static inline int awq_validate_problem_spec(const AwqProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != AWQ_ABI_VERSION) return 0;
    if (spec->max_K < AWQ_MIN_K || spec->max_K > AWQ_MAX_K) return 0;
    if (spec->max_N < AWQ_MIN_N || spec->max_N > AWQ_MAX_N) return 0;
    if (spec->max_G < AWQ_MIN_G || spec->max_G > AWQ_MAX_G) return 0;
    return 1;
}

static inline int awq_validate_run_spec(const AwqRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != AWQ_ABI_VERSION) return 0;
    if (run->K < AWQ_MIN_K || run->K > AWQ_MAX_K) return 0;
    if (run->N < AWQ_MIN_N || run->N > AWQ_MAX_N) return 0;
    if (run->G < AWQ_MIN_G || run->G > AWQ_MAX_G) return 0;
    if (run->G > run->K) return 0;
    if (run->distribution_id < AWQ_DIST_BALANCED ||
        run->distribution_id > AWQ_DIST_RANDBITS) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const AwqProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const AwqProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const AwqRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // AWQ_ACTORDER_REPACK_GEMV_COMMON_H_
