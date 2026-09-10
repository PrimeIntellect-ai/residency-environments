# Task: muon_distopt_stateful_v2

## 1. Task name + one-line summary

**Task name:** muon_distopt_stateful_v2

**One-line summary:** Implement a stateful, simulated-distributed Muon/AdamW optimizer runtime whose persistent momentum/moment buffers, five-step BF16 Newton–Schulz orthogonalization, shape-scaled weight decay updates, logical-rank reduce-scatter/all-gather bookkeeping, and canonical event timeline must match bit-exact FNV-1a-64 checksums across many optimizer steps.

---

## 2. Complete self-contained contract

### 2.1 Agent-visible premise

You implement one `solution.cu` with the v2 C ABI:

```cpp
extern "C" size_t solution_workspace_bytes(const Spec* spec);

extern "C" cudaError_t solution_init(
    const Spec* spec,
    void** state,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const RunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);
```

This is a stateful mini optimizer system, not a pure kernel. Persistent state includes:

- BF16 weights for every parameter.
- FP32 Muon momentum buffers for Muon parameters.
- FP32 Adam first and second moments for Adam parameters.
- Per-parameter step counters and Adam beta-power accumulators.
- Cumulative run/op/invalid/duplicate/nonfinite/update counters.
- A reset counter.

Each `solution_run` is one optimizer step over a batch of logical gradient-shard operations. The batch is first canonically reduced into per-parameter global gradients by a deterministic simulated reduce-scatter. Then every parameter is updated exactly once in ascending `param_id` order. Missing gradients are zero gradients.

The grader compares:

- raw BF16 updated weight bytes,
- FNV checksum over all raw weight bytes,
- FNV checksum over all persistent optimizer state,
- FNV checksum over the canonical per-step event timeline,
- FNV checksum over cumulative counters,
- final aggregate checksum.

No tolerance is used.

---

### 2.2 Common header (`muon_distopt_common.h`)

```cpp
// file: muon_distopt_common.h
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MD_VERSION 2u
#define MD_MAGIC 0x4D554F4E44535432ull  // "MUONDST2"

#define MD_MAX_PARAMS 64u
#define MD_MAX_OPS_PER_RUN 8192u
#define MD_MAX_RANKS 8u

#define MD_DTYPE_BF16 1u

#define MD_PATH_MUON 0u
#define MD_PATH_ADAMW 1u

#define MD_PARAM_FORCE_ADAMW 0x1u
#define MD_PARAM_FORCE_MUON  0x2u

#define MD_STATUS_OK 0u
#define MD_STATUS_BAD_SPEC 1u
#define MD_STATUS_OOM 2u
#define MD_STATUS_BAD_RUN 3u

// FNV constants. This project intentionally does NOT use canonical FNV offset basis.
#define MD_FNV_BASIS 1469598103934665603ull  // hex 0x14650FB0739D0383
#define MD_FNV_PRIME 1099511628211ull

// Float32 bit constants used by the normative algorithm.
#define MD_F32_MUON_A_BITS       0x405C72B0u  // 3.444499969482422f
#define MD_F32_MUON_B_BITS       0xC098CCCDu  // -4.775000095367432f
#define MD_F32_MUON_C_BITS       0x40020419u  // 2.0315001010894775f
#define MD_F32_NS_EPS_BITS       0x33D6BF95u  // 1.0000000116860974e-7f

#define MD_F32_DEFAULT_MOM_BITS  0x3F733333u  // 0.949999988079071f
#define MD_F32_ADAM_B1_BITS      0x3F666666u  // 0.8999999761581421f
#define MD_F32_ADAM_B2_BITS      0x3F7FBE77u  // 0.9990000128746033f
#define MD_F32_ADAM_EPS_BITS     0x322BCC77u  // 9.99999993922529e-9f

struct ParamDesc {
    uint32_t rows;          // >= 1
    uint32_t cols;          // >= 1; numel = rows * cols
    uint32_t storage_dtype; // currently must be MD_DTYPE_BF16
    uint32_t flags;         // FORCE_ADAMW/FORCE_MUON; see path rules below

    // Float32 hyperparameters stored by raw IEEE-754 bits, little-endian.
    // If any field is 0, the default below is used.
    uint32_t lr_bits;             // default 0x3CA3D70A = 0.019999999552965164f
    uint32_t weight_decay_bits;   // default 0x3C23D70A = 0.009999999776482582f
    uint32_t muon_momentum_bits;  // default MD_F32_DEFAULT_MOM_BITS

    uint32_t adam_lr_bits;        // default 0x3A83126F = 0.0010000000474974513f
    uint32_t adam_beta1_bits;     // default MD_F32_ADAM_B1_BITS
    uint32_t adam_beta2_bits;     // default MD_F32_ADAM_B2_BITS
    uint32_t adam_eps_bits;       // default MD_F32_ADAM_EPS_BITS

    uint64_t init_seed;           // deterministic BF16 weight initialization seed
};

struct Spec {
    uint64_t magic;       // must be MD_MAGIC
    uint32_t version;     // must be MD_VERSION
    uint32_t num_params;  // 1..MD_MAX_PARAMS
    uint32_t num_ranks;   // 1..MD_MAX_RANKS
    uint32_t max_ops_per_run;

    uint64_t global_seed;
    uint64_t reserved0;
    uint64_t reserved1;

    ParamDesc params[MD_MAX_PARAMS];
};

struct GradOp {
    uint32_t param_id;       // target parameter
    uint32_t src_rank;       // data-parallel source rank contributing this gradient
    uint32_t dst_rank;       // owner rank for reduce-scatter shard
    uint32_t owner_offset;   // element offset within dst_rank's owner shard
    uint32_t elem_count;     // number of contiguous BF16 grad elements
    uint32_t grad_offset;    // offset into trailing BF16 grad_values[]
    uint32_t flags;          // currently ignored; must be folded into op event
    uint32_t reserved;       // ignored; must be zero in generated tests
};

struct RunSpec {
    uint64_t run_tag;          // arbitrary harness tag, included in timeline
    uint32_t op_count;         // number of GradOp records in inputs
    uint32_t grad_value_count; // number of trailing BF16 values after GradOp array
    uint32_t flags;            // currently 0
    uint32_t reserved;
};

// Device inputs layout:
//   GradOp ops[run->op_count]
//   uint16_t grad_values[run->grad_value_count]
//
// Device outputs layout:
//   RunOutputHeader header
//   uint16_t weights_bf16[total_numel], param-id ascending, row-major within param.
struct RunOutputHeader {
    uint64_t magic;              // MD_MAGIC
    uint32_t version;            // MD_VERSION
    uint32_t status;             // MD_STATUS_*

    uint64_t attempted_runs;     // cumulative, includes OOM/bad-run attempts
    uint64_t successful_runs;    // cumulative successful solution_run calls
    uint64_t reset_count;

    uint64_t run_tag;
    uint64_t weight_hash;
    uint64_t state_hash;
    uint64_t timeline_hash;
    uint64_t counter_hash;
    uint64_t final_hash;

    uint64_t total_valid_ops;
    uint64_t total_invalid_ops;
    uint64_t total_duplicate_ops;
    uint64_t total_nan_inputs;
    uint64_t total_inf_inputs;
    uint64_t total_muon_param_updates;
    uint64_t total_adam_param_updates;

    uint32_t last_valid_ops;
    uint32_t last_invalid_ops;
    uint32_t last_duplicate_ops;
    uint32_t last_nan_inputs;
    uint32_t last_inf_inputs;
    uint32_t last_muon_param_updates;
    uint32_t last_adam_param_updates;
    uint32_t reserved;
};
```

---

### 2.3 Shape family

The starter `solution.cu` is an empty no-op stub. It must not include reference code, derived constants beyond this contract, or any prefilled solution logic.

**Visible calibration family**

- num_params: 6–16.
- num_ranks: 1, 2, 3, or 4.
- Total parameter elements: 1K–64K.
- Matrix Muon params: rows, cols: 4–128. Include square, tall, wide, prime, and ragged dimensions. Examples: 8x8, 7x31, 31x7, 32x96, 97x32, 127x17.
- AdamW params: 1xN, Nx1, and forced-AdamW 2D params. N: 1–4096.
- op_count: 0–2048.
- grad_value_count: 0–128K BF16 values.
- Runs per episode: 3–32.

Includes: missing shards, duplicated ops, out-of-order ops, zero-length ops, NaN/Inf BF16 gradients, rows > cols transpose path, rows <= cols non-transpose path.

**Hidden grading family**

- num_params: 1–64.
- num_ranks: 1–8, including non-power-of-two ranks 3, 5, 7.
- Total parameter elements: 1–262144.
- Matrix Muon params: rows, cols: 1–257. Very skinny/wide shapes such as 1x257, 257x1, 3x251, 251x3.
- AdamW params: 1x1 scalars, 1xN, Nx1, forced-AdamW matrices up to 128K elements.
- op_count: 0–8192. grad_value_count: 0–524288. Runs per episode: 1–128.

Adversarial cases: duplicate ops, overlapping fragments, missing rank contributions, invalid rank IDs, invalid param IDs, range overflow, grad offset overflow, NaN payload canonicalization, +0.0/-0.0 behavior, reset after nontrivial state, insufficient workspace call.

---

### 2.4 Parameter path rule

For parameter p:
- numel = rows * cols.
- If rows == 0 or cols == 0 or rows*cols overflows uint32: bad Spec.
- If storage_dtype != MD_DTYPE_BF16: bad Spec.

**Muon path iff:**
- rows >= 2 AND cols >= 2 AND FORCE_ADAMW flag is not set, OR
- FORCE_MUON flag is set and rows >= 2 and cols >= 2.

**AdamW path otherwise.**

- If both FORCE_ADAMW and FORCE_MUON are set: bad Spec.
- If FORCE_MUON is set on a non-2D param: bad Spec.

---

### 2.5 Initial weights and reset semantics

`solution_init` initializes all weights deterministically from Spec and zeroes all optimizer state.

For parameter `pid`, flattened row-major element `e`, define:

```
x0 = splitmix64(spec.global_seed
                XOR params[pid].init_seed
                XOR 0x9E3779B97F4A7C15 * (pid + 1)
                XOR 0xD1B54A32D192ED03 * (e + 1))

sign = bit 63 of x0
exp_offset = bits [58..60] interpreted as integer 0..7, then mapped to exp = 124 + exp_offset
mant = bits [16..22], 7 bits
bf16_weight_bits = (sign << 15) | (exp << 7) | mant
```

Thus initial weights are always finite normal BF16 numbers with exponents 124..131.

The `splitmix64` function is defined as:
```
x += 0x9E3779B97F4A7C15
x = (x XOR (x >> 30)) * 0xBF58476D1CE4E5B9
x = (x XOR (x >> 27)) * 0x94D049BB133111EB
return x XOR (x >> 31)
```

**Persistent state initialization:**
- Muon momentum[all elements] = +0.0f, raw bits 0x00000000.
- Adam m[all elements] = +0.0f. Adam v[all elements] = +0.0f.
- Adam beta1_pow[pid] = +1.0f, raw bits 0x3F800000.
- Adam beta2_pow[pid] = +1.0f, raw bits 0x3F800000.
- param_step[pid] = 0. All cumulative counters = 0. reset_count = 0.
- attempted_runs = 0. successful_runs = 0.

`solution_reset` restores initial weights and zero optimizer state exactly as above, increments `reset_count` by 1, and preserves no other counters except `reset_count`. After reset:
- attempted_runs = 0, successful_runs = 0
- all op/update/nonfinite counters = 0
- reset_count = old_reset_count + 1

---

### 2.6 BF16 and FP32 exactness

**BF16 storage**

BF16 is stored as a raw little-endian `uint16_t`.

BF16-to-FP32 conversion:
```
f32_bits = uint32_t(bf16_bits) << 16
```

BF16 NaNs are canonicalized on load:
```
if exponent == 0xFF and mantissa != 0:
    f32_bits = 0x7FC00000
```

FP32-to-BF16 conversion uses round-to-nearest-even on the lower 16 discarded bits:
```
if f32 is NaN:
    return 0x7FC0

upper = f32_bits >> 16
lower = f32_bits & 0xFFFF

if lower > 0x8000:
    upper += 1
else if lower == 0x8000 and (upper & 1):
    upper += 1

return uint16_t(upper)
```

Overflow caused by BF16 rounding naturally yields +Inf or -Inf.

**FP32 primitive operations**

All normative FP32 arithmetic is IEEE-754 binary32 round-to-nearest-even. No fused multiply-add contraction is allowed unless explicitly stated; this contract never uses FMA.

Every primitive operation canonicalizes NaN outputs to raw bits 0x7FC00000.

Primitive operations:
- `rn_add(a,b)`: round-to-nearest-even float32 addition, then canonicalize NaN.
- `rn_sub(a,b)`: round-to-nearest-even float32 subtraction, then canonicalize NaN.
- `rn_mul(a,b)`: round-to-nearest-even float32 multiplication, then canonicalize NaN.
- `rn_div(a,b)`: round-to-nearest-even float32 division, then canonicalize NaN.
- `rn_sqrt(a)`: round-to-nearest-even float32 square root, then canonicalize NaN.

Integer-to-float conversions used in shape scale are exact for all shapes in the family.

---

### 2.7 Fixed deterministic reduction tree

All dot products, norms, RMS values, and matrix products use the same canonical pairwise tree.

For a list of K FP32 terms:
```
P = next_power_of_two(K)
terms[K..P-1] = +0.0f

while P > 1:
    for i in 0 .. P/2 - 1:
        terms[i] = rn_add(terms[2*i], terms[2*i + 1])
    P = P/2

result = terms[0]
```

For a dot product of BF16 matrices:
```
term[k] = rn_mul(bf16_to_f32(A[k]), bf16_to_f32(B[k]))
dot = pairwise_tree(term[0..K-1])
```

For Frobenius norm over BF16 matrix X:
```
term[i] = rn_mul(bf16_to_f32(X[i]), bf16_to_f32(X[i]))
sum = pairwise_tree(term[0..numel-1])
norm = rn_sqrt(sum)
```

No atomics, unordered reductions, library GEMMs, or unspecified tensor-core accumulation orders are normative.

---

### 2.8 Simulated logical-rank sharding

For a param with numel elements and R = spec.num_ranks, owner shard of rank r is:
```
shard_start(r) = floor(numel * r / R)
shard_end(r)   = floor(numel * (r + 1) / R)
shard_len(r)   = shard_end(r) - shard_start(r)
```

A GradOp is **valid** iff all are true:
- param_id < num_params
- src_rank < R
- dst_rank < R
- owner_offset <= shard_len(dst_rank)
- elem_count <= shard_len(dst_rank) - owner_offset
- grad_offset <= grad_value_count
- elem_count <= grad_value_count - grad_offset

Zero-length ops can be valid.

**Invalid ops:** contribute no gradient, increment invalid-op counters, are folded into timeline with a deterministic reason code.

Reason codes:
- 1 = bad param_id
- 2 = bad src_rank
- 3 = bad dst_rank
- 4 = owner range overflow
- 5 = grad range overflow
- 6 = reserved/internal

If multiple reasons apply, use the smallest reason code.

**Duplicate op rule:** An op i is duplicate iff it is valid and there exists a valid op j < i with identical: param_id, src_rank, dst_rank, owner_offset, elem_count, grad_offset. Duplicates still contribute gradients. They increment duplicate counters and are folded as duplicate in the op event.

**Reduce-scatter gradient accumulation order:**

For each run, all per-element gradient accumulators are initialized to +0.0f. Then valid ops are accumulated in this exact order:

```
for src_rank in 0 .. R-1:
  for op_index in 0 .. op_count-1:
    if op is valid and op.src_rank == src_rank:
       p = op.param_id
       base = param_base_offset[p]
       local_start = shard_start_p(op.dst_rank) + op.owner_offset

       for k in 0 .. op.elem_count-1:
           raw_bf16 = grad_values[op.grad_offset + k]
           g = canonical_bf16_to_f32(raw_bf16)
           grad_accum[base + local_start + k] =
               rn_add(grad_accum[base + local_start + k], g)
```

---

### 2.9 Muon update semantics

For every Muon parameter p, in ascending param_id, once per successful `solution_run`:

Let:
```
rows = p.rows, cols = p.cols, numel = rows * cols
mu = param.muon_momentum or default 0.95f
lr = param.lr or default 0.02f
wd = param.weight_decay or default 0.01f
a = 3.444499969482422f, b = -4.775000095367432f, c = 2.0315001010894775f
eps = 1.0000000116860974e-7f
```

**Momentum and Nesterov candidate:**

For row-major element e:
```
g = grad_accum[base + e]
m_old = muon_momentum[base + e]
m_new = rn_add(rn_mul(mu, m_old), g)
muon_momentum[base + e] = m_new
candidate[e] = rn_add(g, rn_mul(mu, m_new))  // Nesterov always enabled
```

**Initial BF16 matrix for Newton–Schulz:**

If rows <= cols: M = rows, N = cols; X_pre[i, j] = fp32_to_bf16_rne(candidate[i * cols + j])
If rows > cols: M = cols, N = rows; X_pre[i, j] = fp32_to_bf16_rne(candidate[j * cols + i])

Thus the working matrix X always has M <= N.

Normalize:
```
norm = frobenius_norm_bf16_pairwise(X_pre)
denom = rn_add(norm, eps)
X[i,j] = fp32_to_bf16_rne(rn_div(bf16_to_f32(X_pre[i,j]), denom))
```

**Five quintic Newton–Schulz iterations** (exactly t = 0..4):
```
A  = matmul_bf16_pairwise(X, transpose(X))      // M x M
AA = matmul_bf16_pairwise(A, A)                 // M x M

B[i,j] = fp32_to_bf16_rne(
           rn_add(rn_mul(b, bf16_to_f32(A[i,j])),
                  rn_mul(c, bf16_to_f32(AA[i,j]))))

BX = matmul_bf16_pairwise(B, X)                 // M x N

X_new[i,j] = fp32_to_bf16_rne(
               rn_add(rn_mul(a, bf16_to_f32(X[i,j])),
                      bf16_to_f32(BX[i,j])))

X = X_new
```

After each iteration, compute:
```
iter_norm = frobenius_norm_bf16_pairwise(X)
iter_norm_q = fp32_to_bf16_rne(iter_norm)
```

`iter_norm_q` is folded into the timeline as a `uint16_t`.

**Shape/RMS scaling and weight decay:**

After 5 iterations, orient X back to original shape.
```
shape_scale = rn_sqrt(float(max(rows, cols)))

For original row-major element (r,c):
  if rows <= cols:
    ortho = bf16_to_f32(X[r, c])
  else:
    ortho = bf16_to_f32(X[c, r])

  w_old = bf16_to_f32(weight[base + r*cols + c])
  delta = rn_add(rn_mul(shape_scale, ortho), rn_mul(wd, w_old))
  w_new = rn_sub(w_old, rn_mul(lr, delta))
  weight[base + r*cols + c] = fp32_to_bf16_rne(w_new)

param_step[p] += 1.
```

---

### 2.10 AdamW update semantics

For every AdamW parameter p, in ascending param_id, once per successful `solution_run`:

Let:
```
lr = param.adam_lr or default 0.001f
wd = param.weight_decay or default 0.01f
beta1 = param.adam_beta1 or default 0.9f
beta2 = param.adam_beta2 or default 0.999f
eps = param.adam_eps or default 1e-8f
```

At the start of the param update:
```
param_step[p] += 1
beta1_pow[p] = rn_mul(beta1_pow[p], beta1)
beta2_pow[p] = rn_mul(beta2_pow[p], beta2)
bc1 = rn_sub(1.0f, beta1_pow[p])
bc2 = rn_sub(1.0f, beta2_pow[p])
```

For each element e:
```
g = grad_accum[base + e]
w_old = bf16_to_f32(weight[base + e])
m = adam_m[base + e]; v = adam_v[base + e]

m = rn_add(rn_mul(beta1, m), rn_mul(rn_sub(1.0f, beta1), g))
gg = rn_mul(g, g)
v = rn_add(rn_mul(beta2, v), rn_mul(rn_sub(1.0f, beta2), gg))

adam_m[base + e] = m; adam_v[base + e] = v

mhat = rn_div(m, bc1)
vhat = rn_div(v, bc2)
denom = rn_add(rn_sqrt(vhat), eps)
adam_update = rn_div(mhat, denom)

delta = rn_add(adam_update, rn_mul(wd, w_old))
w_new = rn_sub(w_old, rn_mul(lr, delta))

weight[base + e] = fp32_to_bf16_rne(w_new)
```

---

### 2.11 Nonfinite handling

On input gradient load:
- BF16 NaN → canonical FP32 NaN 0x7FC00000 and increments nan counter.
- BF16 +Inf/-Inf → corresponding FP32 infinity and increments inf counter.

If an input op is duplicate, its nonfinite inputs are still counted again when accumulated.

All FP32 primitive results that are NaN are canonicalized to 0x7FC00000.

All BF16 NaNs stored to weights or temporary BF16 matrices are canonical 0x7FC0.

---

### 2.12 Workspace requirement

`solution_workspace_bytes(spec)` returns:

```
align64(sizeof(float) * total_numel)                    // grad_accum
+ align64(sizeof(float) * next_pow2(max(max_param_numel, max_dim)))
+ align64(sizeof(uint16_t) * max_muon_work_elems * 5)
+ align64(sizeof(uint8_t) * max_ops_per_run)
```

Where:
```
max_param_numel = max(rows * cols)
for each Muon param:
    M = min(rows, cols)
    N = max(rows, cols)
    muon_work_elems = 2*M*N + 3*M*M   // X, BX, A, AA, B
max_muon_work_elems = max over Muon params, or 1 if no Muon params
max_dim = max(rows, cols) over all params
```

If `solution_run` receives `workspace_bytes` smaller than this, it must:
- attempted_runs += 1
- write output header with status MD_STATUS_OOM if outputs != nullptr
- not modify weights or optimizer state
- return cudaErrorMemoryAllocation

---

### 2.13 FNV-1a-64 primitive and field order

The FNV hash primitive is exactly:

```
uint64 h = 1469598103934665603  // 0x14650FB0739D0383
for each byte b in canonical byte stream:
    h = h XOR b
    h = h * 1099511628211 modulo 2^64
```

All integer fields are folded little-endian with fixed widths. BF16 values are folded as little-endian `uint16_t`. FP32 values are folded as raw little-endian `uint32_t` bits after canonical NaN normalization.

**Weight hash** — fold only raw BF16 weight bytes:
```
for pid in 0..num_params-1:
  for e in 0..numel(pid)-1:
    fold_u16(weight[base(pid)+e])
```

**State hash** — field order:
```
fold_u64(MD_MAGIC)
fold_u32(MD_VERSION)
fold_u32(num_params)
fold_u32(num_ranks)

for pid in 0..num_params-1:
    fold_u32(pid)
    fold_u32(path)
    fold_u32(rows)
    fold_u32(cols)
    fold_u64(param_step[pid])

    for e in row-major:
        fold_u16(weight[base+e])

    if path == MUON:
        for e in row-major:
            fold_u32(raw_bits(muon_momentum[base+e]))
    else:
        fold_u32(raw_bits(beta1_pow[pid]))
        fold_u32(raw_bits(beta2_pow[pid]))
        for e in row-major:
            fold_u32(raw_bits(adam_m[base+e]))
        for e in row-major:
            fold_u32(raw_bits(adam_v[base+e]))
```

**Counter hash** — field order:
```
fold_u64(attempted_runs)
fold_u64(successful_runs)
fold_u64(reset_count)
fold_u64(total_valid_ops)
fold_u64(total_invalid_ops)
fold_u64(total_duplicate_ops)
fold_u64(total_nan_inputs)
fold_u64(total_inf_inputs)
fold_u64(total_muon_param_updates)
fold_u64(total_adam_param_updates)
```

**Timeline hash** — fresh per successful run:

Event tags: 0xA0=RUN_BEGIN, 0xA1=OP_ACCEPT, 0xA2=OP_INVALID, 0xA3=PARAM_BEGIN, 0xA4=REDUCE_SCATTER_RANK, 0xA5=MUON_MOMENTUM, 0xA6=MUON_NS_ITER, 0xA7=MUON_APPLY, 0xA8=ADAMW_APPLY, 0xA9=PARAM_END, 0xAA=RUN_END.

```
RUN_BEGIN:
  fold_u8(0xA0)
  fold_u64(run_tag)
  fold_u64(successful_runs_before_this_run)
  fold_u32(op_count)
  fold_u32(grad_value_count)
  fold_u32(num_ranks)

For op_index in original order:
  If valid:
    fold_u8(0xA1); fold_u32(op_index); fold_u32(param_id); fold_u32(src_rank);
    fold_u32(dst_rank); fold_u32(owner_offset); fold_u32(elem_count);
    fold_u32(grad_offset); fold_u32(flags); fold_u32(duplicate ? 1 : 0)
  Else:
    fold_u8(0xA2); fold_u32(op_index); fold_u32(param_id); fold_u32(src_rank);
    fold_u32(dst_rank); fold_u32(owner_offset); fold_u32(elem_count);
    fold_u32(grad_offset); fold_u32(flags); fold_u32(reason_code)

For pid in ascending order:
  PARAM_BEGIN:
    fold_u8(0xA3); fold_u32(pid); fold_u32(path); fold_u32(rows); fold_u32(cols);
    fold_u64(param_step_before)

  For rank r in 0..num_ranks-1:
    rank_grad_hash = FNV over raw FP32 bits of grad_accum[shard_start..shard_end-1]
    REDUCE_SCATTER_RANK:
      fold_u8(0xA4); fold_u32(pid); fold_u32(r); fold_u32(shard_start);
      fold_u32(shard_len); fold_u64(rank_grad_hash)

  If Muon:
    MUON_MOMENTUM:
      fold_u8(0xA5); fold_u32(pid)
      fold_u64(hash of raw FP32 momentum buffer after update)
      fold_u16(fp32_to_bf16_rne(frobenius_norm_bf16_pairwise(X_normalized_initial)))

    For iter t=0..4:
      MUON_NS_ITER:
        fold_u8(0xA6); fold_u32(pid); fold_u32(t); fold_u16(iter_norm_q)
        fold_u64(hash of raw BF16 X after this iteration)

    MUON_APPLY:
      fold_u8(0xA7); fold_u32(pid); fold_u32(raw_bits(shape_scale))
      fold_u64(hash of raw BF16 oriented orthogonal update)
      fold_u64(hash of raw BF16 weights after apply)

  If AdamW:
    ADAMW_APPLY:
      fold_u8(0xA8); fold_u32(pid)
      fold_u32(raw_bits(beta1_pow_after)); fold_u32(raw_bits(beta2_pow_after))
      fold_u64(hash of raw FP32 m after apply)
      fold_u64(hash of raw FP32 v after apply)
      fold_u64(hash of raw BF16 weights after apply)

  PARAM_END:
    fold_u8(0xA9); fold_u32(pid); fold_u64(param_step_after)
    fold_u64(hash of raw BF16 weights after apply)

RUN_END:
  fold_u8(0xAA)
  fold_u32(last_valid_ops); fold_u32(last_invalid_ops); fold_u32(last_duplicate_ops)
  fold_u32(last_nan_inputs); fold_u32(last_inf_inputs)
  fold_u32(last_muon_param_updates); fold_u32(last_adam_param_updates)
  fold_u64(weight_hash); fold_u64(state_hash); fold_u64(counter_hash_after_success)
```

**Final hash:**
```
h = FNV_BASIS
fold_u64(weight_hash)
fold_u64(state_hash)
fold_u64(timeline_hash)
fold_u64(counter_hash)
fold_u64(successful_runs)
fold_u64(reset_count)
```

---

### 2.14 Output-writing order

On successful run:
1. Update internal state.
2. Compute weight_hash.
3. Compute state_hash.
4. Update cumulative counters.
5. Compute counter_hash.
6. Complete timeline with RUN_END, including the hashes.
7. Compute final_hash.
8. Write RunOutputHeader.
9. Copy all raw BF16 weights into outputs after header.

The grader validates both the header fields and the raw weight tensor bytes.

---

## 3. Why this is strictly hard for GPT-5.5

This should be expected to score 0/3 for GPT-5.5-style agentic solvers unless they already have a near-reference implementation strategy, because it couples five failure-prone subsystems that must all match bit-for-bit:

1. **Stateful optimizer semantics across runs.** A solver must persist weights, Muon momentum, AdamW moments, Adam beta powers, per-param step counts, reset behavior, and cumulative counters. A pure-function optimizer or "recompute from inputs" approach fails after the first run.

2. **Two optimizers in one runtime.** Muon and AdamW have different states, update equations, hyperparameters, and event records. Forced-AdamW 2D params and forced-Muon rejection rules prevent simple "2D means Muon" shortcuts.

3. **Exact five-step BF16 Newton–Schulz reproduction.** The task pins BF16 storage after every matrix product and linear combination, FP32 pairwise-tree accumulation, exact coefficients, normalization order, transpose-if-rows>cols, and per-iteration norm events. Mathematically equivalent orthogonalization, library GEMM, tensor-core order, FMA contraction, or different BF16 rounding fails.

4. **Simulated distributed reduce-scatter/all-gather.** Ops are not simply applied in input order. They are validated in input order, but accumulated by src_rank then op_index, constrained by owner-rank shards, with duplicates contributing but also being counted. This is exactly the kind of coordination/bookkeeping that LLM solvers tend to approximate incorrectly.

5. **Canonical timeline and checksum protocol.** Even if weights are correct, the solution fails if timeline FNV fields, state hash fields, counter hash order, invalid reason codes, nonfinite canonicalization, duplicate detection, or little-endian folds differ.

6. **Edge behavior is part of the contract.** Missing shards still produce zero-gradient optimizer steps. Zero-length valid ops exist. NaNs are canonicalized at primitive boundaries. OOM must leave state unchanged. Reset preserves only reset_count. Hidden tests will hit these.

---

## 4. Clarifications (normative)

These clarifications are normative and take precedence over any looser wording above. They were added after cross-validation revealed an under-specified point.

### 4.1 Insufficient-workspace (OOM) output-header hash fields

§2.12 specifies the *side effects* of an insufficient-workspace call: increment `attempted_runs`, leave all weights and optimizer state unchanged, write status `MD_STATUS_OOM` when `outputs != nullptr`, and return `cudaErrorMemoryAllocation`. The exact contents of the OOM `RunOutputHeader` are also part of the contract. When `outputs != nullptr`, the OOM header MUST be written as follows:

- `magic = MD_MAGIC`, `version = MD_VERSION`, `status = MD_STATUS_OOM`.
- `attempted_runs` = the cumulative attempted-run counter *after* this call's `+= 1`. `successful_runs` and `reset_count` = their current (unchanged) values.
- `run_tag = run->run_tag`.
- `weight_hash`, `state_hash`, `counter_hash` = the §2.13 hashes computed over the current, unchanged state. (Because `attempted_runs` was just incremented, the OOM `counter_hash` reflects that increment, consistent with §2.13.)
- `timeline_hash = MD_FNV_BASIS`. No timeline is produced for a rejected run, so this field is a fresh FNV-1a-64 accumulator left at the §2.13 offset basis with no bytes folded.
- `final_hash = MD_FNV_BASIS`. The §2.13 final-hash fold is **not** performed for an OOM run; this field is the same fresh-accumulator value as `timeline_hash`.
- `total_valid_ops`, `total_invalid_ops`, `total_duplicate_ops`, `total_nan_inputs`, `total_inf_inputs`, `total_muon_param_updates`, `total_adam_param_updates` = their current cumulative values.

No weights are copied into the output region after the header on an OOM call. The per-run `last_*` fields and `reserved` are not consumed by the grader for an OOM run.

This field is load-bearing because the grader folds every run's `final_hash` — including OOM runs — into its per-family aggregate checksum. An unspecified OOM `final_hash` makes that aggregate non-reproducible across conforming implementations.
