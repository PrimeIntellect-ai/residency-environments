# Task: stateful_norm_bwd_cachegrad_v2

## 1. Task name and summary

Task name: `stateful_norm_bwd_cachegrad_v2`

One-line summary: Implement a persistent CUDA mini-engine that saves deterministic LN/RMS forward statistics, reuses those cached statistics in mixed LayerNorm/RMSNorm backward ops, computes bit-exact dx, performs deterministic two-stage no-atomic dgamma/dbeta aggregation, accumulates parameter gradients across micro-batches until explicit flushes, and emits exact FNV-scored output/state/timeline snapshots.

---

## 2. Complete self-contained contract

### 2.1 common.h (agent-visible; include verbatim)

```c
// file: common.h
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Stateful PMPP-Pipeline v2 ABI
// -----------------------------------------------------------------------------
struct Spec;
struct RunSpec;

size_t solution_workspace_bytes(const Spec* spec);

cudaError_t solution_init(
    const Spec* spec,
    void** state,
    cudaStream_t stream);

cudaError_t solution_run(
    void* state,
    const RunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

void solution_destroy(void* state);

#ifdef __cplusplus
}
#endif

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
static constexpr uint64_t LNRBWD_MAGIC = 0x4C4E524257443201ull; // "LNRBWD2\1"
static constexpr uint32_t LNRBWD_VERSION = 1;

static constexpr uint32_t LNR_ROWS_PER_PARTIAL = 8;
static constexpr uint32_t LNR_SEGMENT_COLS = 256;
static constexpr uint32_t LNR_MAX_SEGMENTS = 128;       // hidden_size <= 32768
static constexpr uint32_t LNR_STAGE2_LEAVES = 1024;     // max partials per run/op reduction

static constexpr uint64_t LNR_FNV_BASIS = 1469598103934665603ull; // 0x14650FB0739D0383
static constexpr uint64_t LNR_FNV_PRIME = 1099511628211ull;       // 0x00000100000001B3

// -----------------------------------------------------------------------------
// Dtypes
// -----------------------------------------------------------------------------
enum LnrStorageDType : uint32_t {
    LNR_DTYPE_BF16 = 1, // x, dy, gamma, dx are raw IEEE bf16 uint16 payloads
    LNR_DTYPE_F32  = 2  // x, dy, gamma, dx are raw IEEE float32 payloads
};

// -----------------------------------------------------------------------------
// Operation kinds
// -----------------------------------------------------------------------------
enum LnrOpKind : uint8_t {
    LNR_OP_SAVE_LN   = 1, // compute and cache LN mean+rstd from x
    LNR_OP_SAVE_RMS  = 2, // compute and cache RMS rstd from x; cached mean := +0.0f
    LNR_OP_BWD_LN    = 3, // use cached LN mean+rstd, compute dx and accumulate dgamma/dbeta
    LNR_OP_BWD_RMS   = 4, // use cached RMS rstd, compute dx and accumulate dgamma; dbeta += 0
    LNR_OP_FLUSH     = 5  // emit accumulated dgamma/dbeta for one param_id and zero that accumulator
};

enum LnrCacheKind : uint32_t {
    LNR_CACHE_INVALID = 0,
    LNR_CACHE_LN      = 1,
    LNR_CACHE_RMS     = 2
};

enum LnrStatus : uint8_t {
    LNR_STATUS_OK               = 0,
    LNR_STATUS_INVALID_RANGE    = 1,
    LNR_STATUS_INVALID_PARAM    = 2,
    LNR_STATUS_CACHE_MISS       = 3,
    LNR_STATUS_PARTIAL_OVERFLOW = 4,
    LNR_STATUS_UNSUPPORTED_OP   = 5,
    LNR_STATUS_OUTPUT_RANGE     = 6
};

// -----------------------------------------------------------------------------
// Spec: fixed for one solver state.
// -----------------------------------------------------------------------------
#pragma pack(push, 1)

struct Spec {
    uint64_t magic;                    // must be LNRBWD_MAGIC
    uint32_t version;                  // must be LNRBWD_VERSION

    uint32_t hidden_size;              // N, 1 <= N <= 32768
    uint32_t param_count;              // 1 <= param_count <= 16
    uint32_t max_cache_rows;           // 1 <= max_cache_rows <= 65536

    uint32_t max_input_rows_per_run;   // x/dy row capacity visible to a run
    uint32_t max_dx_rows_per_run;      // dx logical output rows visible to a run
    uint32_t max_backward_rows_per_run;// sum of rows over valid backward ops, <= 8192
    uint32_t max_ops_per_run;          // <= 512
    uint32_t max_flush_records_per_run;// <= 512

    uint32_t storage_dtype;            // LNR_DTYPE_BF16 or LNR_DTYPE_F32
    float eps_ln;                      // finite, >= 0
    float eps_rms;                     // finite, >= 0

    uint64_t counter_seed;             // all uint64 counters start here; enables wrap tests

    uint32_t flags;                    // must be 0 in v1
    uint32_t reserved[7];              // must be 0
};

// One op. All ranges are logical rows; element addressing uses strides in RunSpec.
struct OpDesc {
    uint8_t kind;          // LnrOpKind
    uint8_t reserved_kind; // must be 0
    uint16_t flags;        // must be 0 in v1

    uint32_t param_id;     // for BWD/FLUSH; ignored for SAVE
    uint32_t x_row_base;   // source row base in x for SAVE/BWD
    uint32_t dy_row_base;  // source row base in dy for BWD; ignored for SAVE/FLUSH
    uint32_t cache_base;   // first cache slot for SAVE/BWD
    uint32_t rows;         // row count for SAVE/BWD; ignored for FLUSH

    uint32_t dx_out_base;    // first logical dx output row for BWD
    uint32_t flush_out_base; // flush record index for FLUSH
    uint32_t reserved0;      // must be 0
};

// RunSpec lives in host memory. run->ops is a host pointer to op_count OpDesc values.
struct RunSpec {
    uint64_t run_id;

    uint32_t op_count;
    uint32_t input_rows;    // logical row capacity of inputs x/dy for this run
    uint32_t dx_rows;       // logical rows in outputs.dx folded by scorer
    uint32_t flush_records; // logical rows in outputs.flush_dgamma/dbeta folded by scorer

    uint32_t x_stride_elems;     // >= hidden_size
    uint32_t dy_stride_elems;    // >= hidden_size
    uint32_t gamma_stride_elems; // >= hidden_size

    uint32_t reserved0;          // must be 0
    const OpDesc* ops;           // host pointer
};

// Host struct of device pointers.
struct InputPtrs {
    const void* x;     // storage_dtype, shape [input_rows, x_stride_elems]
    const void* dy;    // storage_dtype, shape [input_rows, dy_stride_elems]
    const void* gamma; // storage_dtype, shape [param_count, gamma_stride_elems]
};

// Host struct of device pointers. All snapshots are mandatory and overwritten every run.
struct OutputPtrs {
    void* dx; // storage_dtype, contiguous logical [dx_rows, hidden_size]

    float* flush_dgamma; // fp32 contiguous [flush_records, hidden_size]
    float* flush_dbeta;  // fp32 contiguous [flush_records, hidden_size]

    float* accum_dgamma_snapshot; // fp32 contiguous [param_count, hidden_size], after run
    float* accum_dbeta_snapshot;  // fp32 contiguous [param_count, hidden_size], after run

    float* cache_mean_snapshot;   // fp32 [max_cache_rows], after run
    float* cache_rstd_snapshot;   // fp32 [max_cache_rows], after run
    uint32_t* cache_kind_snapshot;// u32  [max_cache_rows], after run
    uint64_t* cache_gen_snapshot; // u64  [max_cache_rows], after run

    struct TimelineRecord* timeline; // [op_count], packed records below
    struct CounterSnapshot* counters; // [1]
};

// Packed event record. The scorer folds fields in exactly this order and width.
struct TimelineRecord {
    uint64_t run_id;
    uint64_t global_op_index; // cumulative op counter value after increment for this op

    uint32_t op_index_in_run;
    uint8_t kind;
    uint8_t status;
    uint8_t cache_kind_required; // 0 for SAVE/FLUSH, 1 LN, 2 RMS for backward
    uint8_t reserved_a;          // 0

    uint32_t param_id;
    uint32_t rows;
    uint32_t x_row_base;
    uint32_t dy_row_base;
    uint32_t cache_base;
    uint32_t dx_out_base;
    uint32_t flush_out_base;

    uint32_t partial_base;
    uint32_t partial_count;

    uint64_t cache_generation_first; // 0 if not applicable
    uint64_t cache_generation_last;  // 0 if not applicable

    uint64_t counter_snapshot_after_op; // same as global_op_index in v1
};

// All counters are cumulative since init/reset and wrap modulo 2^64.
struct CounterSnapshot {
    uint64_t run_count;
    uint64_t op_count;

    uint64_t save_ln_rows;
    uint64_t save_rms_rows;
    uint64_t bwd_ln_rows;
    uint64_t bwd_rms_rows;

    uint64_t partial_blocks;
    uint64_t flush_count;

    uint64_t invalid_ops;
    uint64_t cache_miss_ops;
    uint64_t cache_overwrite_rows;

    uint64_t cache_generation_counter;
    uint64_t last_run_id;
};

#pragma pack(pop)
```

### 2.2 Shape families

**Visible calibration shapes:**

| Class | hidden_size | dtype | rows/run | ops/run | params | cache rows | Notes |
|-------|-------------|-------|----------|---------|--------|------------|-------|
| V0 | 64 | bf16 | 1–16 | 1–12 | 1 | 64 | simple LN/RMS save→backward→flush |
| V1 | 257 | bf16 | 0–96 | 4–32 | 2 | 512 | ragged column tail, zero-row ops |
| V2 | 1024 | f32 | 16–256 | 8–64 | 3 | 2048 | duplicate params, duplicate dx ranges |
| V3 | 4096 | bf16 | 64–512 | 16–96 | 4 | 4096 | multi-run accumulation before flush |
| V4 | 8193 | bf16 | 64–1024 | 16–128 | 4 | 8192 | ragged segment tail, near max stage-2 reduction |

**Hidden family:**

- `hidden_size` ∈ {1, 2, 3, 7, 16, 31, 64, 127, 128, 129, 255, 256, 257, 511, 512, 769, 1000, 1023, 1024, 1536, 2048, 4095, 4096, 4097, 8191, 8192, 8193, 12288, 16384, 32768}
- `storage_dtype` ∈ {bf16, f32}
- `param_count` ∈ [1, 16]
- `max_backward_rows_per_run` ∈ [0, 8192]
- `op_count` ∈ [0, 512]
- `max_cache_rows` ∈ [1, 65536]
- Input row strides satisfy `stride >= hidden_size`; hidden tests use stride padding in {0, 1, 3, 7, 31, 64, 127} elements.

The solver must not specialize to visible shapes. Hidden tests mix LN/RMS, duplicate param_id, overlapping dx_out_base, cache overwrite, cache miss, zero-variance rows, ±inf, canonical quiet NaNs, row count 0, ragged tails, and counter wrap using `counter_seed ≈ UINT64_MAX`.

### 2.3 Exact numeric semantics

All math is fp32 round-to-nearest-even, no fma contraction, no fast math, no FTZ. The normative primitive operations are:

```
rn_add(a,b)  = IEEE-754 binary32 addition, round-to-nearest-even
rn_sub(a,b)  = IEEE-754 binary32 subtraction, round-to-nearest-even
rn_mul(a,b)  = IEEE-754 binary32 multiplication, round-to-nearest-even
rn_div(a,b)  = IEEE-754 binary32 division, round-to-nearest-even
rn_sqrt(a)   = IEEE-754 binary32 sqrt, round-to-nearest-even
```

Every primitive result is canonicalized: if the result is any NaN, replace it with raw fp32 bits `0x7FC00000`. Input NaNs are canonicalized on load. BF16 output NaNs are stored as `0x7FC0`.

No fma is allowed. For example, `a*b + c` must be `rn_add(rn_mul(a,b), c)`.

### 2.4 BF16 conversion

**BF16 load:**
```
bf16 h -> fp32 bits (uint32(h) << 16), then canonicalize NaN.
```

**BF16 store from fp32:**
```
x = canonicalize_nan(x)
if x is NaN: store 0x7FC0
else:
    u = raw fp32 bits of x
    lsb = (u >> 16) & 1
    bias = 0x7FFF + lsb
    store uint16((u + bias) >> 16)
```

This is round-half-even.

### 2.5 Forward-stat cache semantics

The state contains:
```
accum_dgamma[param_count, hidden_size] : fp32
accum_dbeta [param_count, hidden_size] : fp32

cache_mean[max_cache_rows] : fp32
cache_rstd[max_cache_rows] : fp32
cache_kind[max_cache_rows] : u32, 0 invalid, 1 LN, 2 RMS
cache_gen [max_cache_rows] : u64
CounterSnapshot counters
```

`solution_init` and `solution_reset` zero all arrays, set all `cache_kind` to invalid, set all uint64 counters to `counter_seed`, and set `last_run_id = 0`.

**SAVE_LN**

For each row `r = 0..rows-1`, source row is `x_row_base + r`, cache slot is `cache_base + r`.

LayerNorm saved mean/rstd is computed by deterministic Welford pairwise reduction over `hidden_size` columns.

Leaf for column j:
```
W(j) = { n = 1, mean = load_x(row,j), M2 = +0.0f }
```

Empty padded leaves:
```
W_empty = { n = 0, mean = +0.0f, M2 = +0.0f }
```

Welford combine `C = combine(A,B)`:
```
if A.n == 0: C = B
else if B.n == 0: C = A
else:
    n     = A.n + B.n                  // exact uint32
    delta = rn_sub(B.mean, A.mean)
    ratio = rn_div(float(B.n), float(n))
    mean  = rn_add(A.mean, rn_mul(delta, ratio))

    ab    = rn_mul(float(A.n), float(B.n))
    scale = rn_div(ab, float(n))
    term  = rn_mul(rn_mul(delta, delta), scale)
    M2    = rn_add(rn_add(A.M2, B.M2), term)

    C = {n, mean, M2}
```

**Reduction order:**
1. Partition columns into fixed 256-column segments.
2. Inside each segment, reduce exactly 256 leaves with a binary tree: stride = 1,2,4,...,128; combine `leaf[i]` with `leaf[i+stride]` for `i` divisible by `2*stride`. Missing columns in the final segment are `W_empty`.
3. Reduce exactly 128 segment results with the same binary tree. Missing segments are `W_empty`.

Then:
```
mean = W.mean
var  = rn_div(W.M2, float(hidden_size))          // biased variance
rstd = rn_div(1.0f, rn_sqrt(rn_add(var, eps_ln)))
cache_mean[slot] = mean
cache_rstd[slot] = rstd
cache_kind[slot] = LNR_CACHE_LN
```

**Cache generation:** For each saved row in increasing `r`, first increment `cache_generation_counter` modulo 2^64, then store that value in `cache_gen[slot]`. If the target slot was valid before the save, increment `cache_overwrite_rows`.

**SAVE_RMS**

For each row, compute deterministic sum of squares:
```
leaf_j = rn_mul(x_j, x_j)
```

Reduction order is the same fixed 256-leaf segment tree, then fixed 128-leaf segment tree, padded with `+0.0f`.

Then:
```
ms   = rn_div(sum_sq, float(hidden_size))
rstd = rn_div(1.0f, rn_sqrt(rn_add(ms, eps_rms)))

cache_mean[slot] = +0.0f
cache_rstd[slot] = rstd
cache_kind[slot] = LNR_CACHE_RMS
cache_gen[slot]  = next generation value
```

### 2.6 Backward semantics

A backward op is valid only if:
- `param_id < param_count`
- `x_row_base + rows <= input_rows`
- `dy_row_base + rows <= input_rows`
- `cache_base + rows <= max_cache_rows`
- `dx_out_base + rows <= dx_rows`
- `sum(valid backward rows in this run) <= max_backward_rows_per_run`
- `partial_base + ceil(rows/8) <= ceil(max_backward_rows_per_run/8)`
- all required cache slots are valid and match required kind

If any validation fails, the op is skipped, the requested dx rows are zeroed when the output range is valid, no gradient accumulation occurs, and a non-OK timeline status is emitted.

For valid backward ops, row `r` uses:
```
x row      = x_row_base + r
dy row     = dy_row_base + r
dx row     = dx_out_base + r
cache slot = cache_base + r
gamma row  = param_id
```

Gamma is always required.

**Per-row deterministic reductions for dx**

For each row, compute `c1` and `c2` using exactly the same fixed 256-column segment tree and fixed 128-segment tree.

**For LayerNorm:**
```
mean = cache_mean[slot]
rstd = cache_rstd[slot]

xhat_j = rn_mul(rn_sub(x_j, mean), rstd)
v_j    = rn_mul(dy_j, gamma_j)

c1 leaf = v_j
c2 leaf = rn_mul(v_j, xhat_j)
```
Then:
```
Nf   = float(hidden_size)
invN = rn_div(1.0f, Nf)

term0 = rn_mul(Nf, v_j)
term1 = rn_sub(term0, c1)
term2 = rn_sub(term1, rn_mul(xhat_j, c2))
scale = rn_mul(rstd, invN)

dx_j = rn_mul(scale, term2)
```

**For RMSNorm:**
```
rstd = cache_rstd[slot]

xhat_j = rn_mul(x_j, rstd)
v_j    = rn_mul(dy_j, gamma_j)

c1 leaf = +0.0f
c2 leaf = rn_mul(v_j, xhat_j)
```
Then:
```
Nf    = float(hidden_size)
invN  = rn_div(1.0f, Nf)
coeff = rn_mul(invN, c2)
inner = rn_sub(v_j, rn_mul(xhat_j, coeff))

dx_j = rn_mul(rstd, inner)
```

dx is stored in `storage_dtype`.

### 2.7 Deterministic two-stage dgamma/dbeta

For a valid backward op with `rows = R`:
```
partial_count = ceil(R / 8)
partial_base  = current run-local partial cursor
```

The workspace contains:
```
partial_dgamma[max_partials_per_run, hidden_size] : fp32
partial_dbeta [max_partials_per_run, hidden_size] : fp32
```
where `max_partials_per_run = ceil(max_backward_rows_per_run / 8)`.

**Stage 1: row-block partials**

For each partial block `p = 0..partial_count-1` and column `j`, rows covered are:
```
local row k = p*8 + t, t = 0..7
```

Each partial is reduced with a fixed 8-leaf binary tree, padded with `+0.0f`.

LayerNorm contribution:
```
xhat = rn_mul(rn_sub(x, mean), rstd)
dg_leaf = rn_mul(dy, xhat)
db_leaf = dy
```

RMSNorm contribution:
```
xhat = rn_mul(x, rstd)
dg_leaf = rn_mul(dy, xhat)
db_leaf = +0.0f
```

Store:
```
partial_dgamma[partial_base + p, j] = tree8(dg_leaf)
partial_dbeta [partial_base + p, j] = tree8(db_leaf)
```

**Stage 2: deterministic reduction into persistent accumulators**

For each column `j`, reduce exactly 1024 leaves:
```
leaf[p] = partial_dgamma[partial_base + p, j] for p < partial_count
leaf[p] = +0.0f otherwise
```
using binary-tree strides 1,2,4,...,512. Then:
```
accum_dgamma[param_id,j] = rn_add(accum_dgamma[param_id,j], reduced_dgamma)
accum_dbeta[param_id,j]  = rn_add(accum_dbeta[param_id,j],  reduced_dbeta)
```

This op-order addition is mandatory. If two ops in a run target the same `param_id`, they accumulate in `op_index` order.

**No atomics are permitted.**

### 2.8 Flush semantics

`LNR_OP_FLUSH` validates:
- `param_id < param_count`
- `flush_out_base < flush_records`

If valid:
```
flush_dgamma[flush_out_base,j] = accum_dgamma[param_id,j]
flush_dbeta [flush_out_base,j] = accum_dbeta [param_id,j]
accum_dgamma[param_id,j] = +0.0f
accum_dbeta [param_id,j] = +0.0f
```
for all `j = 0..hidden_size-1`.

If invalid, the op is skipped and the flush output remains zero.

### 2.9 Output initialization and overlapping writes

At the start of every `solution_run`, the solver must zero:
```
dx[dx_rows, hidden_size]
flush_dgamma[flush_records, hidden_size]
flush_dbeta [flush_records, hidden_size]
```

If two valid backward ops write overlapping dx rows, later ops overwrite earlier bytes because ops execute in order. If two valid flush ops write the same `flush_out_base`, later ops overwrite earlier bytes. Snapshots are written after all ops complete.

### 2.10 Timeline and counter semantics

`run_count` increments once at the start of every top-level valid `solution_run`.

For each op in order:
```
op_count increments by 1 modulo 2^64.
TimelineRecord.global_op_index = op_count after increment.
TimelineRecord.counter_snapshot_after_op = same value.
```

For valid saves/backwards/flushes, row/partial/flush counters increment as named.

For invalid ops: `invalid_ops += 1`

For cache misses: `invalid_ops += 1`, `cache_miss_ops += 1`

All uint64 counters wrap modulo 2^64.

### 2.11 Invalid top-level call rules

`solution_init` returns `cudaErrorInvalidValue` if Spec is invalid.

`solution_workspace_bytes(spec)` returns 0 for invalid Spec.

`solution_run` returns `cudaErrorInvalidValue` and performs no state changes if:
- `state == nullptr`
- `run == nullptr`
- `inputs == nullptr`
- `outputs == nullptr`
- `workspace == nullptr` while required workspace bytes > 0
- `workspace_bytes < solution_workspace_bytes(spec)`
- `run->op_count > spec->max_ops_per_run`
- `run->input_rows > spec->max_input_rows_per_run`
- `run->dx_rows > spec->max_dx_rows_per_run`
- `run->flush_records > spec->max_flush_records_per_run`
- x/dy/gamma strides < hidden_size
- `run->ops == nullptr` while `op_count > 0`
- any required output snapshot pointer is null

Per-op invalidity is not a top-level error; it is recorded in the timeline.

### 2.12 FNV scoring contract

The harness computes FNV-1a-64. **Initial hash:**
```
h = 1469598103934665603  (= 0x14650FB0739D0383)
```

**FNV prime:** `1099511628211` (= 0x00000100000001B3)

For each byte b:
```
h = h XOR uint64(b)
h = h * 1099511628211 modulo 2^64
```

All multi-byte scalar fields are folded little-endian with exactly these widths:
- u8 → 1 byte
- u16 → 2 bytes
- u32 → 4 bytes
- u64 → 8 bytes
- fp32 → raw IEEE-754 4 bytes, little-endian
- bf16 → raw 2 bytes, little-endian

**No struct padding is folded.**

For each run, the scorer computes separate hashes:

- **H_dx:** dx logical rows 0..dx_rows-1, cols 0..hidden_size-1, storage_dtype raw bytes.
- **H_flush_dgamma:** flush_records × hidden_size fp32 raw bytes.
- **H_flush_dbeta:** flush_records × hidden_size fp32 raw bytes.
- **H_accum_dgamma:** param_count × hidden_size fp32 raw bytes from `accum_dgamma_snapshot`.
- **H_accum_dbeta:** param_count × hidden_size fp32 raw bytes from `accum_dbeta_snapshot`.
- **H_cache:** `cache_mean_snapshot[max_cache_rows]` fp32, `cache_rstd_snapshot[max_cache_rows]` fp32, `cache_kind_snapshot[max_cache_rows]` u32, `cache_gen_snapshot[max_cache_rows]` u64, in that order.
- **H_timeline:** for op i = 0..op_count-1, fold TimelineRecord fields in common.h order (listed field by field, no struct padding).
- **H_counters:** fold CounterSnapshot fields in common.h order.
- **H_all:** start from basis and fold the eight u64 hashes above in this order: H_dx, H_flush_dgamma, H_flush_dbeta, H_accum_dgamma, H_accum_dbeta, H_cache, H_timeline, H_counters.

A rollout passes only if every run's full tuple matches the reference.

---

## 3. Why this is hard

**Stateful cache reuse is easy to fake wrong.** Backward uses cached mean/rstd, not recomputed stats. Hidden tests deliberately mutate backward x after SAVE_*, overwrite cache slots, mix same-slot LN/RMS generations, and perform backward in later `solution_run` calls.

**The floating-point algorithm is exact, not mathematical.** The solver must reproduce Welford combine order, fixed 256-leaf and 128-segment trees, fixed 8-row partial trees, fixed 1024-leaf stage-2 trees, no fma, fp32 rounding, BF16 round-half-even, NaN canonicalization, signed-zero effects, inf propagation, and op-order accumulator addition.

**Two-stage parameter-gradient aggregation couples layout and arithmetic.** A mathematically equivalent `atomicAdd`, row-major serial sum, CUB reduction, or different block tiling fails because the reduction order and padded leaves are different.

**LN/RMS branches share machinery but differ in subtle terms.** LN uses cached mean, `c1 = sum(dy*gamma)`, `c2 = sum(dy*gamma*xhat)`, and `dbeta += dy`; RMS uses cached rstd, no mean, only `c2`, and `dbeta += 0`. Mixing these branches in one run catches copy-paste mistakes.

**Timeline/counters/state snapshots make partial correctness insufficient.** Even if dx matches, the solver must emit exact event statuses, partial counts, cache generations, counter wrap behavior, accumulator snapshots, and flush ordering.

---
## Clarifications (normative — fold into the contract)
- ABI input/output pointers are device pointers of the dtypes named in the struct field comments.
- `dx` and `flush` output pointers may be null when their corresponding counts are 0 (no write expected).
- `cache_kind`-required scratch must be zero-initialized before use.
- The timeline hash `H_timeline` for an empty run (no ops) folds zero events onto the running basis (base case = unchanged accumulator).
