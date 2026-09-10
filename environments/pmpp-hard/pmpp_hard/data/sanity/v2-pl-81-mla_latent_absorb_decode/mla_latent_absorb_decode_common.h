// ============================================================================
// file: mla_latent_absorb_decode_common.h
// ============================================================================

#ifndef MLA_LATENT_ABSORB_DECODE_COMMON_H_
#define MLA_LATENT_ABSORB_DECODE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MLA_ABI_VERSION 1

#define MLA_MIN_B 1
#define MLA_MAX_B 32
#define MLA_MIN_HQ 1
#define MLA_MAX_HQ 32
#define MLA_MIN_SEQ_CAP 64
#define MLA_MAX_SEQ_CAP 8192
#define MLA_MAX_NEW_TOKENS 8
#define MLA_QUANT_GROUP 32

#define MLA_Y_ATOL 2.5e-3f
#define MLA_Y_RTOL 2.5e-3f
#define MLA_LSE_ATOL 2.5e-3f
#define MLA_LSE_RTOL 2.5e-3f

/*
CONTRACT: mla_latent_absorb_decode

Stateful single-GPU MLA-style (multi-head latent attention) decode pipeline
with an append-only compressed KV cache. Instead of per-head K/V, every
token stores ONE shared latent vector c_t (dimension d_c) and one shared
decoupled positional key r_t (dimension d_r), both quantized to int8 with
per-32-channel power-of-two group scales. Per-head keys/values exist only
implicitly through the persistent up-projection matrices W_uk / W_uv; a fast
implementation absorbs them into the query and the softmax-weighted latent
sum instead of materializing per-token per-head keys or values.

solution_init allocates all persistent state and captures the persistent
weights. solution_run advances exactly one step (chunked prefill and
single-token decode share the entry point: a step appends 1..8 tokens per
active row and emits one attention output per appended token).
solution_reset returns all persistent state (caches, lengths, counters,
running hashes) to the initial configuration; weights are retained.

Legal configuration (MlaProblemSpec):
  B:            number of logical sequences.  1 <= B <= 32.
  Hq:           query heads.                  1 <= Hq <= 32.
  d_c:          latent dimension.             d_c in {128, 192, 256}.
  d_r:          decoupled rope dimension.     d_r in {32, 64}.
  d_h:          per-head query/key dim.       d_h in {64, 128}.
  d_v:          per-head value dim.           d_v in {64, 128}.
  max_seq_len:  hard cap on logical length.   64 <= max_seq_len <= 8192.

Persistent weights (MlaInitInputs, device pointers, fp32):
  W_uk[Hq, d_h, d_c]:  key up-projection per head.
                       element (h, i, j) at W_uk[((h*d_h) + i)*d_c + j].
  W_uv[Hq, d_v, d_c]:  value up-projection per head.
                       element (h, i, j) at W_uv[((h*d_v) + i)*d_c + j].
  rope_cos[max_seq_len, d_r/2], rope_sin[max_seq_len, d_r/2]:
                       position tables, element (t, j) at [t*(d_r/2) + j].
  The pointers are valid only during the solution_init call plus any work
  enqueued on `stream` before it returns; the harness synchronizes `stream`
  after solution_init returns and may free or reuse the buffers afterwards.
  Implementations must copy whatever they need into persistent state.

Persistent state (logical model; representation is free):
  - Quantized latent cache: for every sequence b and position t < seq_len[b],
    the stored group scales and int8 bytes of c_t and r_t (storage format
    below). Append-only; stored bytes never change after the append.
  - seq_len[B] int32: tokens appended so far (never exceeds max_seq_len; the
    harness guarantees inputs never overflow it).
  - step_counter int32: starts at 0, incremented at the START of every
    solution_run (even when active_count == 0).
  - sat_count int32: cumulative count of quantized elements clamped during
    quantization (definition below) since reset.
  - total_tokens int32: cumulative appended tokens since reset.

Storage format (NORMATIVE, byte-exact):
  Every appended token stores, for x = c_t (d_c channels) then r_t (d_r
  channels), per 32-channel group g (channels [32g, 32g+32)):
    - one int8 scale exponent scale_exp,
    - 32 int8 quantized bytes.

  Quantization of one 32-channel fp32 group x[0..32):
    amax = max_d |x[d]|                    (exact fp32 comparisons)
    if amax == 0:
      scale_exp = 0; every byte = 0.
    else:
      kfloor    = floor(log2(amax))        (for normal fp32 this is the
                                            unbiased exponent field; inputs
                                            guarantee amax is normal)
      scale_exp = clamp(kfloor - 6, -110, 110)   stored as int8
      z[d]      = x[d] * 2^(-scale_exp)    (exact: power-of-two scaling;
                                            amax*2^-scale_exp lies in [64,128))
      n[d]      = rne(z[d])                round to NEAREST integer, exact
                                            ties to the EVEN integer
      byte[d]   = clamp(n[d], -127, 127)
      sat_count += 1 for every element with n[d] > 127 or n[d] < -127
      (-128 is never stored).

  Dequantization (NORMATIVE):
    dequant(byte, scale_exp) = byte * 2^scale_exp.

Input guarantees (harness invariants; implementations may rely on them):
  - new_c / new_r entries of REAL token slots (nt < new_token_count[a]) are
    finite fp32, |x| <= 4, and either 0.0f or |x| >= 2^-90 (every nonzero
    group amax is a normal fp32). Padding slots (nt >= new_token_count[a])
    contain arbitrary finite values and must not affect any output.
  - q / q_rope entries of real slots are finite with |x| <= 1.
  - W_uk / W_uv entries are finite with |w| <= 0.25.
  - rope_cos / rope_sin entries are finite in [-1, 1].
  - active_seq entries are unique within a step and in [0, B).
  - new_token_count entries are in [1, MLA_MAX_NEW_TOKENS].
  - seq_len[b] + appended tokens never exceeds max_seq_len.

Per-step RunSpec:
  active_count: rows in this step, 0 <= active_count <= B.
  step_id:      opaque, for debugging only.

Inputs (device pointers):
  active_seq[active_count]                       int32
  new_token_count[active_count]                  int32, in [1, 8]
  new_c[active_count, 8, d_c]                    fp32 (pad slots ignored)
  new_r[active_count, 8, d_r]                    fp32 (pad slots ignored)
  q[active_count, 8, Hq, d_h]                    fp32 (pad slots ignored)
  q_rope[active_count, 8, Hq, d_r]               fp32 (pad slots ignored)

Step semantics (NORMATIVE order):
  (1) step_counter += 1.
  (2) Appends: for every active row a, tokens nt = 0..new_token_count[a]-1
      are appended at positions pos = seq_len_before[seq] + nt (quantize
      new_c[a][nt] and new_r[a][nt] per the storage format);
      seq_len[seq] += new_token_count[a]; total_tokens accumulates.
      Rows touch disjoint sequences, so row order is immaterial.
  (3) Attention (from the post-append cache): for every active row a, real
      token slot nt (position p = seq_len_before + nt), and head hq:
        ql        = W_uk[hq]^T q[a,nt,hq,:]        (vector in R^{d_c};
                    ql[j] = sum_i q[i] * W_uk[hq][i][j])
        for every position t in [0, p]  (the token attends to the whole
        prefix INCLUDING itself; the live set is never empty):
          c~_t    = dequant of stored c bytes of (seq, t)   (d_c values)
          r~_t    = dequant of stored r bytes of (seq, t)   (d_r values)
          rrot_t[2j]   = r~_t[2j]*rope_cos[t][j] - r~_t[2j+1]*rope_sin[t][j]
          rrot_t[2j+1] = r~_t[2j]*rope_sin[t][j] + r~_t[2j+1]*rope_cos[t][j]
          score(t) = ( dot(ql, c~_t) + dot(q_rope[a,nt,hq,:], rrot_t) )
                     * rsqrt(d_h + d_r)
        m  = max_t score(t)
        l  = sum_t exp(score(t) - m)
        z  = sum_t (exp(score(t) - m) / l) * c~_t   (vector in R^{d_c})
        y[a,nt,hq,i]  = sum_j W_uv[hq][i][j] * z[j]
        lse[a,nt,hq]  = m + log(l)                  (natural log)
      Any numerically stable evaluation order is allowed (including
      materializing per-token keys/values instead of absorbing; the absorbed
      order above is what the grader's tolerance is centred on).
      The grader compares |got - expected| <= ATOL + RTOL * |expected|
      with ATOL = RTOL = 2.5e-3 for both y and lse.
  (4) Padding slots nt >= new_token_count[a]: y[a,nt,:,:] and lse[a,nt,:]
      must be written as exact 0.0f.
  (5) Outputs seq_len / cache_hash / meta_checksum / sat_count /
      total_tokens reflect the post-append state and are written on EVERY
      step, including active_count == 0.

FNV-1a-64 PRIMITIVE (NORMATIVE -- this project uses a NON-CANONICAL basis;
  the agent MUST use this exact basis or every checksum fails):
    offset basis = 1469598103934665603  (hex 0x14650FB0739D0383) -- this is NOT
      the canonical FNV-1a-64 basis 14695981039346656037 (0xCBF29CE484222325).
    prime        = 1099511628211  (0x100000001B3).
    fold: start h = offset basis; absorb each field's raw bytes little-endian
      at its full declared width (i8=1, i32=4, u64=8, two's complement); per
      byte b: h = (h XOR b) * prime (mod 2^64). No length prefix / terminator
      / final mix.

Outputs (device pointers):
  y[active_count, 8, Hq, d_v]   fp32   (tolerance above; pad slots exact 0)
  lse[active_count, 8, Hq]      fp32   (tolerance above; pad slots exact 0)
  seq_len[B]                    int32  exact
  cache_hash[B]                 uint64 exact; per sequence b:
      h = basis
      for t = 0..seq_len[b]-1 ascending:
        for g = 0..d_c/32-1:  fold c_scale_exp(b,t,g)   (1 int8 byte)
        for d = 0..d_c-1:     fold c_byte(b,t,d)        (1 int8 byte)
        for g = 0..d_r/32-1:  fold r_scale_exp(b,t,g)   (1 int8 byte)
        for d = 0..d_r-1:     fold r_byte(b,t,d)        (1 int8 byte)
      fold seq_len[b] (int32)
      (the trailing seq_len fold means a running per-sequence hash over the
      append-only token stream can be maintained incrementally and finished
      with one 4-byte fold each step)
  meta_checksum[1]              uint64 exact; global:
      h = basis
      fold as int32 each: B, Hq, d_c, d_r, d_h, d_v, max_seq_len,
                          step_counter
      for b = 0..B-1 ascending: fold seq_len[b] (int32)
      fold sat_count, total_tokens (each int32)
  sat_count[1]                  int32  exact (cumulative since reset)
  total_tokens[1]               int32  exact (cumulative since reset)

Determinism:
  - Exact replay of the same step sequence must reproduce every output bit
    (checksums included) exactly.
  - Permuting the row order within a step must reproduce EVERY output
    exactly (appends touch disjoint sequences and the counters are sums, so
    full order invariance is required, unlike tasks with shared free lists).

Rules for solution_run:
  - May launch kernels and use cudaMemsetAsync / device-to-device async copies.
  - May not call cudaMalloc/cudaFree (allocate persistent state in
    solution_init; persistent allocations up to ~1 GiB are acceptable).
  - May not perform synchronous host/device copies or otherwise synchronize
    the stream inside solution_run.
  - May not use CUB, Thrust, cuBLAS, cuDNN, or CUTLASS.
  - Must use only the provided workspace plus persistent state.
*/

struct alignas(8) MlaProblemSpec {
    int32_t abi_version;
    int32_t B;
    int32_t Hq;
    int32_t d_c;
    int32_t d_r;
    int32_t d_h;
    int32_t d_v;
    int32_t max_seq_len;
    int32_t flags;
    int32_t reserved[7];
};

struct alignas(8) MlaRunSpec {
    int32_t abi_version;
    int32_t active_count;
    int32_t step_id;
    int32_t reserved[13];
};

struct alignas(8) MlaInitInputs {
    const float* W_uk;
    const float* W_uv;
    const float* rope_cos;
    const float* rope_sin;
};

struct alignas(8) MlaInputs {
    const int32_t* active_seq;
    const int32_t* new_token_count;
    const float* new_c;
    const float* new_r;
    const float* q;
    const float* q_rope;
};

struct alignas(8) MlaOutputs {
    float* y;
    float* lse;
    int32_t* seq_len;
    uint64_t* cache_hash;
    uint64_t* meta_checksum;
    int32_t* sat_count;
    int32_t* total_tokens;
};

static_assert(sizeof(MlaProblemSpec) == 64, "MlaProblemSpec layout drift");
static_assert(sizeof(MlaRunSpec) == 64, "MlaRunSpec layout drift");
static_assert(sizeof(MlaInitInputs) == 32, "MlaInitInputs layout drift");
static_assert(sizeof(MlaInputs) == 48, "MlaInputs layout drift");
static_assert(sizeof(MlaOutputs) == 56, "MlaOutputs layout drift");

static_assert(offsetof(MlaProblemSpec, abi_version) == 0, "layout");
static_assert(offsetof(MlaProblemSpec, B) == 4, "layout");
static_assert(offsetof(MlaProblemSpec, Hq) == 8, "layout");
static_assert(offsetof(MlaProblemSpec, d_c) == 12, "layout");
static_assert(offsetof(MlaProblemSpec, d_r) == 16, "layout");
static_assert(offsetof(MlaProblemSpec, d_h) == 20, "layout");
static_assert(offsetof(MlaProblemSpec, d_v) == 24, "layout");
static_assert(offsetof(MlaProblemSpec, max_seq_len) == 28, "layout");

static_assert(offsetof(MlaRunSpec, abi_version) == 0, "layout");
static_assert(offsetof(MlaRunSpec, active_count) == 4, "layout");
static_assert(offsetof(MlaRunSpec, step_id) == 8, "layout");

static inline size_t mla_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int mla_valid_dc(int d_c) {
    return d_c == 128 || d_c == 192 || d_c == 256;
}

static inline int mla_valid_dr(int d_r) {
    return d_r == 32 || d_r == 64;
}

static inline int mla_valid_dh(int d_h) {
    return d_h == 64 || d_h == 128;
}

static inline int mla_valid_dv(int d_v) {
    return d_v == 64 || d_v == 128;
}

static inline int mla_validate_problem_spec(const MlaProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MLA_ABI_VERSION) return 0;
    if (spec->B < MLA_MIN_B || spec->B > MLA_MAX_B) return 0;
    if (spec->Hq < MLA_MIN_HQ || spec->Hq > MLA_MAX_HQ) return 0;
    if (!mla_valid_dc(spec->d_c)) return 0;
    if (!mla_valid_dr(spec->d_r)) return 0;
    if (!mla_valid_dh(spec->d_h)) return 0;
    if (!mla_valid_dv(spec->d_v)) return 0;
    if (spec->max_seq_len < MLA_MIN_SEQ_CAP || spec->max_seq_len > MLA_MAX_SEQ_CAP) return 0;
    return 1;
}

static inline int mla_validate_run_spec(const MlaRunSpec* run, const MlaProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != MLA_ABI_VERSION) return 0;
    if (run->active_count < 0 || run->active_count > spec->B) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const MlaProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const MlaProblemSpec* spec,
    const MlaInitInputs* init_inputs,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const MlaRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // MLA_LATENT_ABSORB_DECODE_COMMON_H_
