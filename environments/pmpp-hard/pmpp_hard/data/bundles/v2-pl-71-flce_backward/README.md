# FLCB-StateTrain: Stateful Fused Linear Cross-Entropy Backward Engine

## 1. Task Summary

A stateful CUDA mini-training engine that processes micro-batches through a deterministic chunked fused linear + cross-entropy backward pass, accumulates dW/d_bias persistently across `solution_run` calls, emits bit-exact dX, per-row losses, flush snapshots, event timelines, and cumulative counters, and is graded only by exact FNV-1a-64 checksums.

Grounding: this is deliberately modeled after Liger-Kernel's fused linear cross-entropy pattern: chunk tokens to avoid full BT × V materialization, compute CE gradients in-place during forward, then derive grad_input, grad_weight, and grad_bias from those chunked gradients. Liger's paper describes operation fusion and input chunking as core memory/performance techniques, and its implementation explicitly computes chunked logits, calls a CE kernel that overwrites logits with gradients, then accumulates grad_input, grad_weight, and grad_bias. PyTorch's public CE contract fixes the relevant ignore_index and label_smoothing semantics: ignored class-index targets do not contribute to loss or input gradient, and label smoothing mixes the hard target with a uniform distribution.

---

## 2. Complete Self-Contained Contract

### 2.1 Agent-Visible ABI: common.h

```c
// common.h — agent-visible ABI for FLCB-StateTrain v1.
//
// The starter solution.cu shipped to solvers MUST be an empty no-op ABI stub.
// It must not contain reference code, helper kernels, hidden constants, or
// checksum answers.

#pragma once
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#define FLCB_MAGIC   0x464C4342u      // "FLCB"
#define FLCB_VERSION 1u

// Project FNV basis, intentionally NOT canonical FNV offset.
#define FLCB_FNV_BASIS 1469598103934665603ull  // 0x14650FB0739D0383
#define FLCB_FNV_PRIME 1099511628211ull

enum FLCBOpCode : uint32_t {
    FLCB_OP_MICRO = 1,   // process a micro-batch row range
    FLCB_OP_FLUSH = 2,   // emit accumulated dW/dbias snapshot, optionally zero
    FLCB_OP_BUMP  = 3    // adversarial counter bump for wrap testing
};

enum FLCBReduction : uint32_t {
    FLCB_RED_SUM        = 0, // grad_scale = op.loss_scale
    FLCB_RED_MEAN_VALID = 1  // grad_scale = op.loss_scale / valid_count; zero if valid_count == 0
};

enum FLCBFlushMode : uint32_t {
    FLCB_FLUSH_SNAPSHOT      = 0, // emit current accumulators, keep state
    FLCB_FLUSH_EMIT_AND_ZERO = 1  // emit current accumulators, then zero accumulators
};

enum FLCBBumpCounter : uint32_t {
    FLCB_BUMP_ROWS_TOTAL         = 0,
    FLCB_BUMP_VALID_ROWS_TOTAL   = 1,
    FLCB_BUMP_VOCAB_TILE_UPDATES = 2,
    FLCB_BUMP_EVENTS_TOTAL       = 3
};

enum FLCBEventType : uint32_t {
    FLCB_EV_RUN_BEGIN  = 1,
    FLCB_EV_MICRO_BEGIN= 2,
    FLCB_EV_CHUNK_DONE = 3,
    FLCB_EV_MICRO_DONE = 4,
    FLCB_EV_FLUSH      = 5,
    FLCB_EV_BUMP       = 6,
    FLCB_EV_INVALID_OP = 7,
    FLCB_EV_RUN_END    = 8
};

enum FLCBInvalidFlag : uint32_t {
    FLCB_BAD_OPCODE    = 1u << 0,
    FLCB_BAD_RANGE     = 1u << 1,
    FLCB_BAD_EPS       = 1u << 2,
    FLCB_BAD_REDUCTION = 1u << 3,
    FLCB_BAD_FLUSH     = 1u << 4,
    FLCB_BAD_BUMP      = 1u << 5,
    FLCB_BAD_CAPACITY  = 1u << 6
};

struct alignas(8) FLCBSpec {
    uint32_t magic;
    uint32_t version;
    uint32_t H;
    uint32_t V;
    uint32_t max_run_rows;
    uint32_t max_ops;
    uint32_t row_chunk;
    uint32_t vocab_tile;
    int32_t  ignore_index;
    uint32_t has_bias;
    uint32_t max_flushes_per_run;
    uint32_t flags;          // must be 0 for v1
    uint32_t reserved[8];    // must be 0
};

struct alignas(8) FLCBRunSpec {
    uint64_t run_id;
    uint32_t input_rows;
    uint32_t output_row_capacity;
    uint32_t op_count;
    uint32_t event_capacity;
    uint32_t flush_capacity;
    uint32_t flags;          // must be 0 for v1
    uint32_t reserved[4];    // must be 0
};

struct alignas(8) FLCBOp {
    uint32_t opcode;
    uint32_t row_offset;
    uint32_t row_count;
    uint32_t reduction;
    uint32_t label_smoothing_q16; // epsilon = q / 65536.0, valid q <= 65536
    uint32_t loss_scale_bits;     // raw IEEE-754 binary32 upstream scalar
    uint32_t aux;
    uint32_t reserved0;           // must be 0
    uint64_t bump_amount;
    uint64_t tag;
};

struct alignas(8) FLCBInputs {
    const FLCBOp*   ops;       // device array [run.op_count]
    const uint16_t* x_bf16;   // device [run.input_rows, H], row-major BF16 bits
    const int32_t*  target;   // device [run.input_rows]
    const uint16_t* w_bf16;   // device [V, H], row-major BF16 bits
    const uint16_t* bias_bf16;// device [V] if spec.has_bias != 0, else may be null
};

struct alignas(8) FLCBEvent {
    uint32_t type;
    uint32_t op_index;
    uint32_t chunk_index; // 0xffffffff when not applicable
    uint32_t flags;
    uint64_t run_id;
    uint64_t tag;
    uint64_t row_begin;
    uint64_t row_count;
    uint64_t valid_count;
    uint64_t ignored_count;
    uint64_t invalid_count;
    uint64_t counter_rows_after;
    uint64_t counter_valid_after;
    uint64_t flush_generation_after;
    uint64_t event_serial;
};

struct alignas(8) FLCBCounters {
    uint64_t runs;
    uint64_t ops_seen;
    uint64_t micro_ops;
    uint64_t flush_ops;
    uint64_t invalid_ops;
    uint64_t row_chunks;
    uint64_t vocab_tile_updates;
    uint64_t rows_total;
    uint64_t valid_rows_total;
    uint64_t ignored_rows_total;
    uint64_t invalid_targets_total;
    uint64_t output_rows_total;
    uint64_t flush_generation;
    uint64_t events_total;
    uint64_t last_run_id;
    uint64_t reserved[8]; // must stay 0
};

struct alignas(8) FLCBRunReport {
    uint32_t status;
    uint32_t output_rows;
    uint32_t flushes_written;
    uint32_t events_written;
    uint64_t required_workspace_bytes;
    uint64_t reserved[5]; // must stay 0
};

struct alignas(8) FLCBOutputs {
    uint16_t*      dx_bf16;
    float*         loss_f32;
    float*         flush_dW_f32;
    float*         flush_dbias_f32;
    FLCBEvent*     events;
    FLCBCounters*  counters;
    FLCBRunReport* report;
};

extern "C" size_t solution_workspace_bytes(const FLCBSpec* spec);

extern "C" cudaError_t solution_init(
    const FLCBSpec* spec,
    void** state,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const FLCBRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream);

extern "C" void solution_destroy(void* state);
```

Required struct sizes for v1 (static_assert these):

```
sizeof(FLCBSpec)      == 80
sizeof(FLCBRunSpec)   == 48
sizeof(FLCBOp)        == 48
sizeof(FLCBEvent)     == 104
sizeof(FLCBCounters)  == 184
sizeof(FLCBRunReport) == 64
```

If a compiler produces different sizes, the submission is invalid.

### 2.2 Shape Family

**Visible calibration family:**

- H ∈ {16, 31, 32, 63, 64, 96, 128}
- V ∈ {7, 31, 127, 256, 509, 1023, 4096}
- row_chunk ∈ {1, 3, 7, 16, 33, 64}
- vocab_tile ∈ {1, 5, 17, 64, 97, 128}
- input_rows ∈ [0, 2048]
- op_count ∈ [1, 8]
- flush_capacity ∈ [0, op_count]
- label_smoothing_q16 ∈ {0, 3277, 6554, 16384, 32768}
- loss_scale_bits values include +1.0f, -1.0f, 0.25f, 2.0f, -0.0f
- ignore_index = -100 in most visible tests, with some ignore_index ∈ {0, V-1, -1}

**Hidden family:**

- H ∈ [1, 384], not necessarily divisible by 2, 8, 16, or 32
- V ∈ [2, 8192], including primes and V % vocab_tile != 0
- row_chunk ∈ [1, 128]
- vocab_tile ∈ [1, 257]
- input_rows ∈ [0, 8192]
- op_count ∈ [1, 16]
- solution_run sequence length per test case: 1..24
- repeated row ranges are allowed; overlapping micro ops are allowed
- multiple flushes per solution_run are allowed
- OP_BUMP may force selected counters near 2^64 - 1 so wrap behavior is tested

**Dtypes:**
- X, W, and bias: raw BF16 bits, stored as uint16_t, row-major
- dX: raw BF16 bits
- loss, dW, d_bias: binary32 float
- Persistent accumulators: binary32 float
- No FP16, TF32, tensor cores, stochastic rounding, atomics, or fast math in v1 contract

### 2.3 Structural Validity and Error Rules

`solution_workspace_bytes(spec)` returns zero if spec == nullptr or structurally invalid.

A spec is structurally valid iff:
- magic == FLCB_MAGIC
- version == FLCB_VERSION
- H >= 1, V >= 2, max_run_rows >= 0, max_ops >= 1, row_chunk >= 1, vocab_tile >= 1
- has_bias ∈ {0, 1}
- flags == 0
- all reserved fields are zero

**Required workspace:**

```
required_workspace_bytes = align_up(2 * row_chunk * V * sizeof(float), 256)
```

Workspace layout: `float logits[row_chunk, V]` followed by `float dlogits[row_chunk, V]`.

If `solution_run` receives insufficient workspace, null required pointers, op_count > spec.max_ops, input_rows > spec.max_run_rows, nonzero reserved fields, or inadequate output/event/flush capacity, it must return `cudaErrorInvalidValue` before modifying persistent state or output bytes.

Invalid ops inside an otherwise structurally valid run do NOT return an error. They emit `FLCB_EV_INVALID_OP`, increment `invalid_ops`, increment `ops_seen`, and continue.

An op is invalid if:
- unknown opcode
- MICRO range overflows or exceeds run.input_rows
- MICRO.label_smoothing_q16 > 65536
- MICRO.reduction not in {FLCB_RED_SUM, FLCB_RED_MEAN_VALID}
- FLUSH.aux not in {FLCB_FLUSH_SNAPSHOT, FLCB_FLUSH_EMIT_AND_ZERO}
- BUMP.aux not in {0,1,2,3}

### 2.4 BF16 and Floating-Point Primitives

**BF16 decode:** For input BF16 bit pattern b:
```
u32 = uint32_t(b) << 16
f = bitcast_float(u32)
```
Sanitize:
- if exponent(b) == 0xff and mantissa(b) != 0: return +0.0f  (NaN → zero)
- if b == +Inf: return +16.0f
- if b == -Inf: return -16.0f
- otherwise: return f exactly

Finite subnormals and signed zeros are preserved.

**BF16 encode, round-half-even:** For output BF16 from binary32 f:
```
if isnan(f): return 0x7fc0
u = bitcast_uint32(f)
lsb = (u >> 16) & 1
round_bias = 0x7fff + lsb
return uint16_t((u + round_bias) >> 16)
```
+0.0f becomes 0x0000; -0.0f becomes 0x8000.

**Scalar parameter sanitize:** loss_scale_bits interpreted as raw binary32, then:
- NaN → +0.0f
- +Inf → +1.0f
- -Inf → -1.0f
- finite → unchanged

**Floating-point operations:** Every add, subtract, multiply, divide is binary32 round-to-nearest-even. No FMA contraction allowed. Compile with: `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`. `expf` and `logf` are CUDA device functions under the same compilation mode.

### 2.5 Exact 32-Lane Reduction Primitive

For values a[0..N-1], `reduce32_sum(a, N)`:

```
lane_partial[l] = +0.0f for l = 0..31

for l in 0..31:
    for i = l; i < N; i += 32:
        lane_partial[l] = rn_add(lane_partial[l], a[i])

for stride in [16, 8, 4, 2, 1]:
    for l in 0..stride-1:
        lane_partial[l] = rn_add(lane_partial[l], lane_partial[l + stride])

return lane_partial[0]
```

For max with tie-break, `reduce32_argmax_lowest_index(value[i], class_index[i], N)`:
- Higher value wins
- If values exactly equal, lower class_index wins
- Inactive lanes start as (-Inf, UINT32_MAX)
- Same [16,8,4,2,1] tree is used

**This reduction order is part of the checksum contract.** A mathematically equivalent but differently ordered reduction fails.

### 2.6 Exact Logit Computation

For row r, class v:
```
prod[h] = rn_mul(decode_bf16(X[r, h]), decode_bf16(W[v, h]))
dot     = reduce32_sum(prod, H)
logit   = dot                          if has_bias == 0
logit   = rn_add(dot, decode_bf16(bias[v])) if has_bias == 1
```
X and W are never modified. Ignored and invalid target rows skip all logit math and emit zero loss, zero dlogits, and zero dX.

### 2.7 Target Classification

For target y:
```
ignored = (y == ignore_index)
valid   = (!ignored && 0 <= y && y < V)
invalid = (!ignored && !valid)
```
Invalid targets are treated like ignored rows for math. They are counted separately in `invalid_targets_total`. If `ignore_index` is inside [0,V), that class index is still ignored.

### 2.8 Chunked Online Softmax Semantics

For every valid row, classes are processed in vocab tiles. `num_vocab_tiles = ceil_div(V, vocab_tile)`.

Initialize: `m = -Inf`, `s = +0.0f`, `sum_logits = +0.0f`, `pred_class = 0`, `pred_logit = -Inf`.

For tile t:
```
begin = t * vocab_tile
end   = min(V, begin + vocab_tile)
```

Compute all logits in [begin,end) with the exact logit rule.

```
tile_max, tile_argmax = reduce32_argmax_lowest_index(logit[v], v, tile_size)
tile_exp_sum = reduce32_sum(expf(rn_sub(logit[v], tile_max)), tile_size)
tile_logit_sum = reduce32_sum(logit[v], tile_size)
```

Online combine:
```
new_m = max(m, tile_max); if exactly equal, keep old m
left  = rn_mul(s, expf(rn_sub(m, new_m)))   // if m == -Inf, this is +0.0f
right = rn_mul(tile_exp_sum, expf(rn_sub(tile_max, new_m)))
s     = rn_add(left, right)
m     = new_m
sum_logits = rn_add(sum_logits, tile_logit_sum)
```

Prediction tie-break:
```
if tile_max > pred_logit:
    pred_logit = tile_max; pred_class = tile_argmax
else if tile_max == pred_logit:
    pred_class = min(pred_class, tile_argmax)
```

After all tiles:
```
lse = rn_add(m, logf(s))
mean_logit = rn_div(sum_logits, float(V))
```

**This exact online rescale is a central trap.** A solver that computes full-row logsumexp in a different order fails.

### 2.9 Loss Semantics

For valid row target y:
```
eps = float(label_smoothing_q16) / 65536.0f
one_minus_eps = rn_sub(1.0f, eps)

nll_loss    = rn_sub(lse, target_logit)
smooth_loss = rn_sub(lse, mean_logit)

row_loss = rn_add(
    rn_mul(one_minus_eps, nll_loss),
    rn_mul(eps, smooth_loss))
```

For ignored or invalid rows: `row_loss = +0.0f`.

`outputs.loss_f32` stores the unreduced, unscaled row loss for each appended micro row.

### 2.10 dlogits Semantics

For valid row and class v:
```
prob = expf(rn_sub(logit[v], lse))

target_prob =
    eps / float(V)
    + (v == y ? one_minus_eps : +0.0f)

raw_dlogit = rn_sub(prob, target_prob)
```

Gradient scale:
```
if reduction == FLCB_RED_SUM:
    grad_scale = sanitize(loss_scale_bits)

if reduction == FLCB_RED_MEAN_VALID:
    if valid_count_in_this_micro_op == 0:
        grad_scale = +0.0f
    else:
        grad_scale = rn_div(sanitize(loss_scale_bits), float(valid_count_in_this_micro_op))
```

Then: `dlogit = rn_mul(grad_scale, raw_dlogit)`.

For ignored or invalid rows: `dlogit[v] = +0.0f` for all v.

### 2.11 dX Semantics

For every appended row r_out (corresponding to input row r_in) and hidden index h:
```
term[v] = rn_mul(dlogit[v], decode_bf16(W[v, h]))
dx_f32  = reduce32_sum(term, V)
dx_bf16 = encode_bf16_rne(dx_f32)
```

For ignored and invalid rows: `dx_bf16 = 0x0000` for all h.

Output layout is append-only in op order: for each valid MICRO op, append rows `row_offset .. row_offset + row_count - 1`. Repeated and overlapping row ranges append repeated outputs.

### 2.12 Persistent dW and d_bias Semantics

State contains: `acc_dW_f32[V,H]`, `acc_dbias_f32[V]`, counters, flush_generation.

At init and reset, accumulators are all +0.0f.

For each valid MICRO, rows are processed in row chunks (`chunk_begin = 0, row_chunk, 2*row_chunk, ...`). For each chunk, after dlogits are computed:

```
for v in 0..V-1:
  for h in 0..H-1:
      term[i] = rn_mul(dlogit[chunk_row_i, v], decode_bf16(X[input_row_i, h]))
      chunk_grad = reduce32_sum(term, chunk_rows)
      acc_dW[v, h] = rn_add(acc_dW[v, h], chunk_grad)

for v in 0..V-1:
      chunk_bias = reduce32_sum(dlogit[chunk_row_i, v], chunk_rows)
      acc_dbias[v] = rn_add(acc_dbias[v], chunk_bias)
```

Update order: solution_run order → op order → row chunk order → v ascending → h ascending for dW. No float atomics. If `spec.has_bias == 0`, `acc_dbias` still exists and remains zero.

### 2.13 Flush Semantics

For valid FLUSH op:
```
outputs.flush_dW_f32[flush_index, :, :]  = current acc_dW_f32
outputs.flush_dbias_f32[flush_index, :]  = current acc_dbias_f32
flushes_written += 1; flush_ops += 1
```

If `aux == FLCB_FLUSH_EMIT_AND_ZERO`:
```
acc_dW_f32 = +0.0f; acc_dbias_f32 = +0.0f
flush_generation = flush_generation + 1 mod 2^64
```

If `aux == FLCB_FLUSH_SNAPSHOT`: accumulators and flush_generation unchanged.

### 2.14 Counter Wrap

All uint64_t counters wrap modulo 2^64. `OP_BUMP` adds `bump_amount` modulo 2^64 to the selected counter. `events_total` bump happens before the bump event is appended, so the bump event serial includes the wrapped value plus one.

### 2.15 Event Timeline

For each valid structural `solution_run`:
1. Increment `counters.runs`. Set `counters.last_run_id = run_id`.
2. Emit `RUN_BEGIN`.
3. For every op, increment `ops_seen` before processing it.
4. For valid MICRO: increment `micro_ops`. Count valid/ignored/invalid targets for whole op. Emit `MICRO_BEGIN`. For each row chunk: process math, increment `row_chunks`, increment `vocab_tile_updates += chunk_rows * ceil_div(V, vocab_tile)`, emit `CHUNK_DONE`. After all chunks: increment `rows_total += row_count`, `valid_rows_total += valid_count`, `ignored_rows_total += ignored_count`, `invalid_targets_total += invalid_count`, `output_rows_total += row_count`. Emit `MICRO_DONE`.
5. For valid FLUSH: execute flush. Emit `FLUSH`.
6. For valid BUMP: execute bump. Emit `BUMP`.
7. For invalid op: increment `invalid_ops`. Emit `INVALID_OP` with flags set to invalid-reason bitset.
8. At end, emit `RUN_END`.

Every emitted event increments `events_total` first, then writes that value into `event_serial`.

### 2.16 FNV-1a-64 Scoring Primitive

```c
uint64_t h = 1469598103934665603ull; // 0x14650FB0739D0383 — project FNV basis
for each byte b in canonical byte stream:
    h ^= uint64_t(b);
    h *= 1099511628211ull; // modulo 2^64
```

Little-endian field folding: uint32_t = 4 bytes LSB-first; uint64_t = 8 bytes LSB-first; float = 4 bytes of raw IEEE-754 bits; uint16_t BF16 = low byte then high byte. Structs: fold in-memory bytes under required v1 layout; reserved bytes must be zero.

**Scoring streams:**

```
tensor_hash:
    dx_bf16       [report.output_rows, H]               raw bytes
    loss_f32      [report.output_rows]                  raw bytes
    flush_dW_f32  [report.flushes_written, V, H]        raw bytes
    flush_dbias   [report.flushes_written, V]           raw bytes

event_hash:
    events        [report.events_written]               sizeof(FLCBEvent) bytes each

counter_hash:
    counters      [1]                                   sizeof(FLCBCounters) bytes

report_hash:
    report        [1]                                   sizeof(FLCBRunReport) bytes

combined_hash:
    tensor_hash as u64 little-endian
    event_hash as u64 little-endian
    counter_hash as u64 little-endian
    report_hash as u64 little-endian
```

A submission passes a case iff all four component hashes and the combined hash match the reference.

---

## 3. Why This Is Strictly Hard

1. **Exact online-softmax rescale at vocab-tile boundaries.** The solver must reproduce the m/s online combine order exactly, including ragged vocab_tile, tile-level max/sum order, and tie-breaking. Full-logit softmax, PyTorch CE, CUB reductions, or a different tree fail.

2. **Gradient-at-forward coupling.** dlogits are generated from the same chunked logits, label smoothing, reduction denominator, ignore/invalid-row rules, and loss scale. A solver that computes loss and backward in separate "equivalent" passes usually changes float order.

3. **Persistent accumulation state.** dW/d_bias persist across `solution_run` calls until a flush-zero op. Flush snapshots, snapshot-without-zero, repeated row ranges, and overlapping micro ops all interact with event counters.

4. **No atomics, exact reduction trees.** dW is a state update but cannot use float atomics. Each (v,h) update must reduce rows in the 32-lane primitive and then add to persistent state exactly once per row chunk.

5. **Timeline and counter exactness.** Even if numeric tensors are right, the submission must emit the exact event sequence, event serials, chunk counts, invalid-op events, ignored/invalid target counts, flush generation, and modulo-2^64 counter wrap.

6. **BF16 edge behavior.** NaN/Inf sanitization, negative zero, BF16 round-half-even, row skips, and output-byte hashing turn small "reasonable" differences into total failure.

---
## Clarifications (normative — fold into the contract)
- **§2.10 target_prob (CRITICAL — exact float-op order)**: the smoothing term is computed as
  `target_prob = rn_mul(eps, rn_div(1.0f, float(V))) + (v == y ? one_minus_eps : +0.0f)`.
  That is, compute `inv_v = rn_div(1.0f, float(V))` first, then `rn_mul(eps, inv_v)` — NOT `rn_div(eps, float(V))`. The two differ by up to 1 ULP and change every downstream hash when `label_smoothing_q16 > 0`.
- **grad_scale**: `rn_div(sanitize(loss_scale_bits), float(valid_count))` in IEEE-754 round-to-nearest-even (may be computed host-side; default x86_64 SSE RNE matches GPU `--prec-div=true`).
- **`max_flushes_per_run`**: advisory for pre-allocation only; `solution_run` does NOT reject runs on it (it only enforces `need_flushes <= run->flush_capacity`).
- **§2.15 event-record field payload (NORMATIVE — every `FLCBEvent` byte is hashed by `event_hash`)**: §2.15 fixes the event *sequence* and the counter math, but the `event_hash` stream (§2.16) folds the full `sizeof(FLCBEvent)` bytes of each record, so **every field of every emitted event is part of the contract**. The exact per-field payload for each event type is pinned below. All fields not listed for a given type are `0`. `chunk_index` is `0xffffffff` except on `CHUNK_DONE` (where it is the zero-based chunk index `ci`). `op_index` is `0xffffffff` on `RUN_BEGIN`/`RUN_END` and the emitting op's zero-based index on all other events.

  | event type | `flags` | `tag` | `row_begin` | `row_count` | `valid_count` / `ignored_count` / `invalid_count` |
  |---|---|---|---|---|---|
  | `RUN_BEGIN` | `0` | `0` | `0` | `0` | `0` / `0` / `0` |
  | `MICRO_BEGIN` | `0` | `op.tag` | `op.row_offset` | `op.row_count` | op-level valid / ignored / invalid counts |
  | `CHUNK_DONE` | `0` | `op.tag` | `op.row_offset + chunk_begin` | rows in this chunk | this-chunk valid / ignored / invalid counts |
  | `MICRO_DONE` | `0` | `op.tag` | `op.row_offset` | `op.row_count` | op-level valid / ignored / invalid counts |
  | `FLUSH` | **`op.aux`** (the `FLCBFlushMode`: `0`=SNAPSHOT, `1`=EMIT_AND_ZERO) | `op.tag` | `0` | `0` | `0` / `0` / `0` |
  | `BUMP` | **`op.aux`** (the `FLCBBumpCounter`: `0..3`) | `op.tag` | `0` | **`op.bump_amount`** | `0` / `0` / `0` |
  | `INVALID_OP` | invalid-reason `FLCBInvalidFlag` bitset | `op.tag` | `op.row_offset` | `op.row_count` | `0` / `0` / `0` |
  | `RUN_END` | `0` | `0` | `0` | `0` | `0` / `0` / `0` |

  Two encodings that an independent reading commonly gets wrong (each one changes `event_hash` and the cascaded `combined_hash`, while leaving `tensor_hash`/`counter_hash`/`report_hash` correct):
  - **`FLUSH.flags` is `op.aux`, not `0`.** Because SNAPSHOT has `aux == 0`, an implementation that hard-codes `flags = 0` still passes every SNAPSHOT flush and only diverges on an EMIT_AND_ZERO flush.
  - **`BUMP.row_count` is `op.bump_amount`, not `0`, and `BUMP.flags` is `op.aux`, not `0`.** The bump magnitude is carried in the event's `row_count` field; `row_begin` stays `0`. (`BUMP.flags == 0` happens to pass any visible case whose bump selector is `ROWS_TOTAL == 0`, but diverges for selectors `1..3`.)

- **§2.15 `run_id`, `counter_*_after`, `flush_generation_after`, `event_serial` (NORMATIVE)**: for *every* emitted event, `run_id = run.run_id`. The "after" snapshot fields record the **live counter values at the instant the event record is appended** — i.e. *after* any counter mutation this op has already performed:
  - `event_serial` = `events_total` after its own pre-increment for this event (§2.15 final paragraph).
  - `counter_rows_after` = `rows_total`, `counter_valid_after` = `valid_rows_total`, `flush_generation_after` = `flush_generation`, each as of append time. Fixed by the execute-then-emit ordering of §2.15 steps 4–6: a `BUMP` to `rows_total`/`valid_rows_total` is reflected in that same BUMP event's `counter_*_after`; an EMIT_AND_ZERO `FLUSH` increments `flush_generation` **before** emitting its own `FLUSH` event, so that event (and every later event in the run) carries the post-increment generation. Within a MICRO op the op-level `rows_total`/`valid_rows_total` increments land *after* the last `CHUNK_DONE` and *before* `MICRO_DONE`, so `MICRO_BEGIN` and `CHUNK_DONE` see the pre-op totals and `MICRO_DONE` sees the post-op totals.
