# Task: dcrs_allreduce_stream_v2

## 1. Task name + one-line summary

**Task name:** dcrs_allreduce_stream_v2 — Deterministic Compensated Reduction + Sharded All-Reduce Streaming Engine

A stateful CUDA mini-runtime that performs bit-reproducible compensated reductions over logical ranks, decomposes all-reduce into reduce-scatter + all-gather, supports sum / segmented sum / norm modes, and persists running (sum, compensation) accumulator state across solution_run calls.

---

## 2. Complete self-contained contract

### 2.1 ABI

The solver implements exactly one `solution.cu` exposing:

```cpp
extern "C" size_t solution_workspace_bytes(const DcrSpec* spec);

extern "C" cudaError_t solution_init(
    const DcrSpec* spec,
    void** state,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const DcrRunSpec* run,
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

`state` is opaque to the harness and persists across calls to `solution_run`. `solution_reset` restores the exact post-`solution_init` state: all accumulators zero, all accumulator modes unset, all cumulative counters zero, run index zero.

The starter `solution.cu` must be an empty no-op ABI stub. It must not contain any reference logic.

### 2.2 Constants

```cpp
#define DCR_ABI_VERSION 2u

#define DCR_MAX_OPS_PER_RUN 64u
#define DCR_TILE_ITEMS 256u
#define DCR_WARP_SIZE 32u

#define DCR_MODE_SUM  0u
#define DCR_MODE_SEG  1u
#define DCR_MODE_NORM 2u

#define DCR_DTYPE_F32  0u
#define DCR_DTYPE_BF16 1u

#define DCR_OUT_F32  0u
#define DCR_OUT_BF16 1u

#define DCR_CLASS_FINITE  0u
#define DCR_CLASS_POS_INF 1u
#define DCR_CLASS_NEG_INF 2u
#define DCR_CLASS_NAN     3u

#define DCR_STATUS_OK              0u
#define DCR_STATUS_INVALID_OP      1u
#define DCR_STATUS_MODE_MISMATCH   2u
#define DCR_STATUS_OOB_INPUT       3u
#define DCR_STATUS_OOB_OUTPUT      4u
#define DCR_STATUS_UNSUPPORTED     5u

#define DCR_INVALID_F32_BITS 0x7FC0BAD1u
#define DCR_CANONICAL_NAN_F32_BITS 0x7FC00000u
#define DCR_CANONICAL_BF16_NAN 0x7FC0u
#define DCR_INVALID_BF16 0x7FC1u
```

### 2.3 FNV-1a-64 primitive

This task uses the **project FNV basis**, not the canonical FNV basis:

```cpp
DCR_FNV_BASIS = 1469598103934665603ull
              = 0x14650FB0739D0383ull

DCR_FNV_PRIME = 1099511628211ull
              = 0x00000100000001B3ull
```

Byte fold:

```
h = DCR_FNV_BASIS
for each byte b in stream order:
    h = h XOR uint64(b)
    h = (h * DCR_FNV_PRIME) modulo 2^64
```

All integer fields are folded little-endian, least-significant byte first. Floating-point values are folded by their raw IEEE-754 storage bits, not by numeric value.

The final run output contains five hashes:

```
fnv_tensor   = FNV(raw tensor payload bytes only)
fnv_timeline = FNV(canonical timeline bytes only)
fnv_state    = FNV(canonical exported state-record bytes only)
fnv_counters = FNV(canonical cumulative counter bytes only)

fnv_all      = FNV(
                 u32 tag 0x54454E53  // "TENS"
                 u64 tensor_bytes
                 tensor bytes
                 u32 tag 0x54494D45  // "TIME"
                 u64 timeline_bytes
                 timeline bytes
                 u32 tag 0x53544154  // "STAT"
                 u64 state_bytes
                 state bytes
                 u32 tag 0x434E5452  // "CNTR"
                 u64 counter_bytes
                 counter bytes
               )
```

The output header itself is not folded into any FNV hash.

### 2.4 Common structs (verbatim — also provided as `dcrs_allreduce_common.h`)

All structs below are packed little-endian. No padding bytes are part of any canonical stream.

```cpp
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define DCR_ABI_VERSION 2u
#define DCR_MAX_OPS_PER_RUN 64u

struct DcrSpec {
    uint32_t abi_version;          // must be DCR_ABI_VERSION
    uint32_t max_accumulators;     // visible <= 16, hidden <= 128
    uint32_t max_ranks;            // visible <= 8, hidden <= 16
    uint32_t max_cells;            // vector cells or segment count
    uint32_t max_items_per_rank;   // per-rank items M
    uint32_t max_ops_per_run;      // <= DCR_MAX_OPS_PER_RUN
    uint32_t allow_bf16;           // 0 or 1
    uint32_t reserved0;

    uint64_t max_input_bytes;
    uint64_t max_output_bytes;
    uint64_t flags;                // must be zero in v2
};

struct DcrOpDesc {
    uint32_t mode;             // SUM, SEG, NORM, or invalid
    uint32_t input_dtype;      // F32 or BF16
    uint32_t output_dtype;     // F32 or BF16
    uint32_t acc_id;           // persistent accumulator slot

    uint32_t logical_ranks;    // R
    uint32_t cells;            // SUM/NORM vector length, SEG segment count
    uint32_t items_per_rank;   // M
    uint32_t segment_count;    // must equal cells for SEG, zero otherwise

    uint64_t values_offset;    // byte offset into inputs
    uint64_t keys_offset;      // byte offset into inputs, SEG only
    uint64_t tensor_offset;    // byte offset into outputs
    uint64_t state_offset;     // byte offset into outputs

    uint64_t reserved0;
    uint64_t reserved1;
};

struct DcrRunSpec {
    uint32_t abi_version;
    uint32_t op_count;

    uint64_t input_bytes;
    uint64_t output_bytes;

    uint64_t header_offset;          // normally 0
    uint64_t tensor_region_offset;
    uint64_t tensor_region_bytes;
    uint64_t timeline_region_offset;
    uint64_t timeline_region_bytes;  // op_count * sizeof(DcrTimelineRecord)
    uint64_t state_region_offset;
    uint64_t state_region_bytes;
    uint64_t counters_offset;
    uint64_t counters_bytes;

    DcrOpDesc ops[DCR_MAX_OPS_PER_RUN];
};

struct DcrOutputHeader {
    uint64_t magic;          // 0x44535253414C4C32 = "DSRSALL2"
    uint32_t abi_version;
    uint32_t run_status;

    uint64_t run_index_before;
    uint64_t run_index_after;

    uint32_t op_count;
    uint32_t reserved0;

    uint64_t tensor_bytes;
    uint64_t timeline_bytes;
    uint64_t state_bytes;
    uint64_t counters_bytes;

    uint64_t fnv_tensor;
    uint64_t fnv_timeline;
    uint64_t fnv_state;
    uint64_t fnv_counters;
    uint64_t fnv_all;
};

struct DcrTimelineRecord {
    uint64_t run_index;
    uint32_t op_index;
    uint32_t status;

    uint32_t mode;
    uint32_t input_dtype;
    uint32_t output_dtype;
    uint32_t acc_id;

    uint32_t logical_ranks;
    uint32_t cells;
    uint32_t items_per_rank;
    uint32_t segment_count;

    uint32_t tile_items;              // always 256
    uint32_t warp_size;               // always 32
    uint32_t cross_rank_policy;       // always 1: owner-rotated order
    uint32_t shard_policy;            // always 1: contiguous balanced shards

    uint64_t values_offset;
    uint64_t keys_offset;
    uint64_t tensor_offset;
    uint64_t state_offset;
};

struct DcrStateRecord {
    uint32_t acc_id;
    uint32_t cell;
    uint32_t mode;
    uint32_t cls;

    uint32_t sum_bits;
    uint32_t comp_bits;
    uint32_t value_bits;      // finalized running value, f32 bits
    uint32_t reserved0;

    uint64_t update_count;
    uint64_t finite_count;
    uint64_t nan_count;
    uint64_t pos_inf_count;
    uint64_t neg_inf_count;
};

struct DcrCountersRecord {
    uint64_t magic;            // 0x44535253434E5452 = "DSRSCNTR"
    uint32_t abi_version;
    uint32_t reserved0;

    uint64_t run_count;
    uint64_t op_count;
    uint64_t valid_op_count;
    uint64_t invalid_op_count;

    uint64_t finite_input_count;
    uint64_t nan_input_count;
    uint64_t pos_inf_input_count;
    uint64_t neg_inf_input_count;

    uint64_t tensor_bytes_emitted;
    uint64_t state_records_emitted;

    uint64_t fnv_basis;        // 1469598103934665603
    uint64_t fnv_prime;        // 1099511628211
};
```

### 2.5 Shape family

**Visible calibration family:**

| Parameter | Values |
|---|---|
| max_accumulators | 2, 4, 8, 16 |
| logical_ranks R | 1, 2, 4, 8 |
| cells | 1, 2, 7, 31, 128, 1024, 4096 |
| items_per_rank M | 0, 1, 2, 17, 33, 255, 256, 257, 513, 4096 |
| op_count/run | 1..8 |
| solution_run calls | 1..5 |
| input_dtype | f32, bf16 |
| output_dtype | f32, bf16 |
| modes | SUM, SEG, NORM |

**Hidden family:**

| Parameter | Values |
|---|---|
| max_accumulators | up to 128 |
| logical_ranks R | 3, 5, 6, 7, 9, 12, 16 |
| cells | 0, 1, 3, 5, 63, 511, 1023, 8191, 32768 |
| items_per_rank M | 0..65536 |
| op_count/run | 1..64 |
| solution_run calls | 1..32 |
| input_dtype | mixed f32/bf16 in same run |
| output_dtype | mixed f32/bf16 in same run |
| segments | dense, sparse, empty, duplicate-heavy, adversarial key order |
| special values | ±0, subnormals, max finite, overflow-producing finite sums, +Inf, -Inf, quiet NaNs, signaling-NaN bit patterns |

### 2.6 Input layouts

For every op, `values_offset` points into inputs.

**SUM and NORM value layout**

For R = logical_ranks, M = items_per_rank, C = cells:

```
values[r][m][c]

linear element index = ((r * M + m) * C + c)
```

If `input_dtype == DCR_DTYPE_F32`, each element is 4 raw bytes.

If `input_dtype == DCR_DTYPE_BF16`, each element is 2 raw bytes. BF16 to FP32 conversion is exact bit widening:

```cpp
uint32_t fp32_bits = uint32_t(bf16_bits) << 16;
```

**SEG value/key layout**

For R = logical_ranks, M = items_per_rank, S = segment_count = cells:

```
values[r][m]
keys[r][m]
```

`keys_offset` points to R × M little-endian uint32_t keys.

A key contributes to segment s iff:

```
key == s
```

Keys >= segment_count are ignored, not invalid. Duplicate keys are legal. Their order is the original item order m = 0..M-1.

### 2.7 Output tensor layout

Each op writes one all-gathered tensor payload at `outputs + tensor_offset`.

Let:

```
B = output element byte size:
    4 for DCR_OUT_F32
    2 for DCR_OUT_BF16
```

Payload shape is always: **logical_ranks × cells** (rank-major):

```
out_rank[r][c]
linear = r * cells + c
```

All ranks receive identical all-gathered reduced values. Therefore, for a valid op:

```
out_rank[0][c] == out_rank[1][c] == ... == out_rank[R-1][c]
```

The repeated all-gather copies are intentionally included in the raw tensor FNV.

### 2.8 Floating-point environment

All finite arithmetic is IEEE-754 binary32, round-to-nearest-even.

Required CUDA compilation behavior:

- no `--use_fast_math`
- `-ftz=false`
- `-prec-div=true`
- `-prec-sqrt=true`
- `-fmad=false` for this task's arithmetic helpers

Subnormals are not flushed. +0.0f and -0.0f inputs are accepted. Any contracted primitive whose result compares equal to zero is canonicalized to +0.0f.

### 2.9 Canonical compensated pair

Every partial and persistent accumulator cell is:

```
Pair = {
    cls: FINITE | POS_INF | NEG_INF | NAN
    sum:  f32
    comp: f32
    finite_count:  u64
    nan_count:     u64
    pos_inf_count: u64
    neg_inf_count: u64
}
```

**Empty pair:**

```
cls = FINITE
sum = +0.0f
comp = +0.0f
all counts = 0
```

**Leaf construction**

For SUM and SEG:

```
leaf(x):
    if x is NaN:  cls=NAN,     nan_count=1
    if x is +Inf: cls=POS_INF, pos_inf_count=1
    if x is -Inf: cls=NEG_INF, neg_inf_count=1
    else:         cls=FINITE,  sum=canonical_zero(x), comp=+0, finite_count=1
```

For NORM:

```
leaf_norm(x):
    if x is NaN:        cls=NAN, nan_count=1
    if x is +Inf/-Inf:  cls=POS_INF, pos_inf_count=1
    else:
        y = rn_f32(x * x)
        if y is +Inf: cls=POS_INF, pos_inf_count=1
        else:         cls=FINITE, sum=canonical_zero(y), comp=+0, finite_count=1
```

-Inf squared becomes +Inf.

**Neumaier add of finite scalar**

Given finite pair a and finite scalar x:

```
t = rn_f32(a.sum + x)

if t is +Inf or -Inf:
    a.cls = POS_INF or NEG_INF according to sign(t)
    a.sum = t
    a.comp = +0.0f
    return

if abs(a.sum) >= abs(x):
    delta = rn_f32(rn_f32(a.sum - t) + x)
else:
    delta = rn_f32(rn_f32(x - t) + a.sum)

a.comp = rn_f32(a.comp + delta)
a.sum  = t

if a.sum  == 0.0f: a.sum  = +0.0f
if a.comp == 0.0f: a.comp = +0.0f
```

**Pair merge**

`merge(A, B)` mutates A.

Special class handling happens before finite arithmetic:

```
A.nan_count     += B.nan_count       modulo 2^64
A.pos_inf_count += B.pos_inf_count   modulo 2^64
A.neg_inf_count += B.neg_inf_count   modulo 2^64
A.finite_count  += B.finite_count    modulo 2^64

if A.cls == NAN or B.cls == NAN:
    A.cls = NAN
    A.sum = canonical NaN 0x7FC00000
    A.comp = +0
    return

if any positive infinity has been seen and any negative infinity has been seen:
    A.cls = NAN
    A.sum = canonical NaN 0x7FC00000
    A.comp = +0
    return

if any positive infinity has been seen:
    A.cls = POS_INF
    A.sum = +Inf
    A.comp = +0
    return

if any negative infinity has been seen:
    A.cls = NEG_INF
    A.sum = -Inf
    A.comp = +0
    return
```

If both are finite:

```
A = neumaier_add(A, B.sum)
A = neumaier_add(A, B.comp)
```

This exact two-add merge is used everywhere: local tile merge, second-pass merge, cross-rank merge, and streaming state merge.

### 2.10 Canonical fixed reduction tree

The local per-rank reduction is deliberately not "sum in index order." It is the following fixed CUDA tree.

For every rank r, cell/segment c, and tile q:

```
tile_start = q * 256
thread t owns item m = tile_start + t
```

If m >= M, the thread starts with an empty pair.

For SUM/NORM, every active item contributes to cell c. For SEG, item m contributes only if `keys[r][m] == c`; otherwise the thread starts with an empty pair.

**Within-warp tree**

For each warp independently:

```
delta = 16, 8, 4, 2, 1

if lane < delta:
    pair[lane] = merge(pair[lane], pair[lane + delta])
```

The left operand is always the lower lane.

**Block tree over warp results**

There are 8 warps in the 256-thread block. Warp w writes its lane-0 pair to shared slot w.

Warp 0 then reduces slots 0..7 using:

```
delta = 4, 2, 1

if lane < delta:
    pair[lane] = merge(pair[lane], pair[lane + delta])
```

The block result is warp 0 lane 0.

**Second pass over tile partials**

For each (rank, cell) pair, tile partials are reduced by one 256-thread block using the same warp/block tree. Thread t owns tile partial t if t < ceil(M / 256); otherwise it owns an empty pair.

items_per_rank <= 65536, so ceil(M/256) <= 256; exactly one second-pass block is sufficient.

### 2.11 Sharded all-reduce semantics

Let C = cells, R = logical_ranks.

The reduce-scatter owner of cell c is defined by balanced contiguous partitioning:

```
base = C / R
rem  = C % R

rank owner o owns:
    begin(o) = o * base + min(o, rem)
    end(o)   = begin(o) + base + (o < rem ? 1 : 0)

owner(c) is the unique o with begin(o) <= c < end(o)
```

Empty shards are legal when C < R.

For each cell c, only owner(c) performs the conceptual reduce-scatter reduction. The cross-rank sequence is owner-rotated:

```
rank_seq[j] = (owner(c) + j) % R, for j = 0..R-1
```

The R rank partial pairs are then reduced using the same warp-style tree over the logical sequence rank_seq:

```
delta = largest power of two <= R/2, then halve to 1
if logical_lane < delta and logical_lane + delta < R:
    pair[lane] = merge(pair[lane], pair[lane + delta])
```

For non-power-of-two R, missing lanes are skipped.

All-gather then copies the owner result to every output rank in rank-major order.

### 2.12 Mode semantics

**SUM**

For each output cell c:

```
rank_partial[r,c] = deterministic compensated reduction of values[r][0..M-1][c]
step_pair[c]      = deterministic owner-rotated cross-rank merge of rank_partial[*,c]
output[c]         = finalize_sum(step_pair[c])
state[acc_id,c]   = merge(state[acc_id,c], step_pair[c])
```

**SEG**

For each segment s:

```
rank_partial[r,s] = deterministic compensated reduction of values[r][m] where keys[r][m] == s
step_pair[s]      = deterministic owner-rotated cross-rank merge of rank_partial[*,s]
output[s]         = finalize_sum(step_pair[s])
state[acc_id,s]   = merge(state[acc_id,s], step_pair[s])
```

Empty segments produce +0.0f.

**NORM**

For each cell c:

```
rank_partial[r,c] = deterministic compensated reduction of square(values[r][m][c])
step_pair[c]      = deterministic owner-rotated cross-rank merge of rank_partial[*,c]
output[c]         = finalize_norm(step_pair[c])
state[acc_id,c]   = merge(state[acc_id,c], step_pair[c])  // running sum-of-squares pair
```

finalize_norm:

```
if cls == NAN:     canonical NaN 0x7FC00000
if cls == POS_INF: +Inf
if cls == NEG_INF: canonical NaN 0x7FC00000  // should only occur via corrupted state
if finite:
    total = rn_f32(sum + comp)
    if total < +0.0f: total = +0.0f
    result = sqrt.rn.f32(total)
    if result == 0.0f: result = +0.0f
```

### 2.13 Output rounding

finalize_sum:

```
if cls == NAN:     0x7FC00000
if cls == POS_INF: +Inf
if cls == NEG_INF: -Inf
if finite:
    result = rn_f32(sum + comp)
    if result == 0.0f: result = +0.0f
```

If `output_dtype == DCR_OUT_F32`, store raw uint32_t bits little-endian.

If `output_dtype == DCR_OUT_BF16`, convert the finalized FP32 bits to BF16 using round-to-nearest-even:

```cpp
uint32_t lsb = (bits >> 16) & 1u;
uint32_t rounded = bits + 0x7FFFu + lsb;
uint16_t bf16 = uint16_t(rounded >> 16);
```

NaNs are canonicalized before BF16 conversion:

```
f32 NaN output:  0x7FC00000
bf16 NaN output: 0x7FC0
```

Invalid-op poison:

```
f32 poison:  0x7FC0BAD1
bf16 poison: 0x7FC1
```

### 2.14 Persistent state rules

Each acc_id has a mode lock.

Initial mode: `UNSET = 0xFFFFFFFF`

On the first valid op touching an acc_id, the mode lock becomes that op's mode.

A later valid op touching the same acc_id with a different mode is invalid with status `DCR_STATUS_MODE_MISMATCH`, emits poison tensor bytes, writes a timeline record, does not update state, and increments `invalid_op_count`.

For each valid op, after the state update, the solver exports C `DcrStateRecord`s at `outputs + state_offset`, in increasing cell order.

For SUM and SEG, `value_bits` is `finalize_sum(running_state[acc_id, cell])`.
For NORM, `value_bits` is `finalize_norm(running_state[acc_id, cell])`.

Counters wrap modulo 2^64.

### 2.15 Invalid and edge rules

A physically malformed RunSpec may return `cudaErrorInvalidValue`. Hidden graded cases do not depend on undefined memory behavior.

Semantic invalid ops are graded and must be handled.

**Invalid if any condition is true:**

- mode not in {SUM, SEG, NORM}
- input_dtype invalid
- output_dtype invalid
- input_dtype == BF16 and spec.allow_bf16 == 0
- acc_id >= spec.max_accumulators
- logical_ranks == 0 or logical_ranks > spec.max_ranks
- cells > spec.max_cells
- items_per_rank > spec.max_items_per_rank
- op_count > spec.max_ops_per_run
- SEG and segment_count != cells
- non-SEG and segment_count != 0
- mode mismatch for acc_id
- input byte range exceeds run.input_bytes
- output byte range exceeds run.output_bytes

**For a semantic invalid op:**

```
effective_R = clamp(logical_ranks, 1, spec.max_ranks)
effective_C = min(cells, spec.max_cells)
payload size = effective_R * effective_C * output_element_bytes
```

- payload is filled with invalid poison
- timeline status records the reason
- no state records are emitted for that op
- persistent state is unchanged

If effective_C == 0, tensor payload is empty but the invalid timeline record is still emitted.

---

## 3. Why this is strictly hard

This task is hard because it couples multiple exactness traps that are easy to implement "mathematically correctly" but not byte-correctly.

**The reduction tree is weird on purpose.** A solver that sums linearly, uses CUB without matching the exact tree, uses float atomics, or uses a standard block reduction order will fail. The lane-0 order after delta=16,8,4,2,1 is not item order.

**Compensation is stateful and pair-based.** The merge is not just sum += partial. Every merge is add(B.sum) then add(B.comp), with Neumaier compensation and zero canonicalization after each primitive.

**Cross-rank order depends on reduce-scatter owner.** The same cell reduced under a different owner has a different rank order. Non-power-of-two ranks in hidden tests break ring/allreduce assumptions.

**Three modes share one engine but have different leaf semantics.** SUM and SEG reduce values; NORM reduces squared values and finalizes with sqrt.rn.f32. SEG has duplicate keys, ignored keys, empty segments, and arbitrary key order.

**Persistent state is graded.** A solution can produce correct current-step tensors but fail exported running (sum, comp) state after several solution_run calls.

**Special values are not delegated to CUDA accident.** NaN, ±Inf, overflow-to-inf, subnormals, signed zero, BF16 RNE, and invalid-op poison are all specified. Approximate "works on normal floats" kernels fail hidden cases.

---
## Clarifications (normative — fold into the contract)
- **Output buffer region alignment**: all region offsets in the output buffer must be aligned to at least 8 bytes (align8). "No padding bytes in canonical stream" refers to struct fields, not region offsets; the canonical FNV stream still covers the documented region order.
- **`DcrCountersRecord.run_count` semantics**: cumulative run index since init/reset (set to the post-increment run index `run_after`), NOT a per-run valid-op count.
- **SEG-mode input counters**: `finite_input_count` (and the other input-class counters) in SEG mode count ONLY items whose key is in range (`key < segment_count`, i.e. segment-contributing); out-of-range/ignored keys are NOT counted. This differs from SUM/NORM modes, which count every `R*M*C` element.
