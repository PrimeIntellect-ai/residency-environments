# DSG-MoE-BWD: Deterministic Segmented Gradient Scatter for Embedding + MoE Backward

## 1. Task Summary

A stateful CUDA mini-engine that replays MoE routing with capacity overflow, computes exact token input gradients, deterministically segment-reduces duplicate embedding-row gradients and per-expert weight gradients, persists accumulators across micro-batches, and emits exact FNV-checked state/timeline/tensor bytes.

---

## 2. Complete Self-Contained Contract

### 2.1 ABI

Submit **`solution.cu`** implementing exactly these five functions:

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

State persists across `solution_run` calls. `solution_reset` restores the same state as immediately after `solution_init`: zero accumulated gradients, zero counters, run id 0, cumulative event hash equal to the FNV basis, same deterministic expert weights.

### 2.2 Constants

```cpp
static constexpr uint32_t DSG_SPEC_MAGIC = 0x44534753u; // "SGSD" little-endian marker
static constexpr uint32_t DSG_RUN_MAGIC  = 0x44534752u; // "RGSD"
static constexpr uint32_t DSG_OUT_MAGIC  = 0x4453474Fu; // "OGSD"
static constexpr uint32_t DSG_VERSION    = 1;

static constexpr uint32_t DSG_DTYPE_BF16 = 0;
static constexpr uint32_t DSG_DTYPE_F32  = 1;

static constexpr uint32_t DSG_FLAG_EMIT_TABLE        = 1u << 0;
static constexpr uint32_t DSG_FLAG_EMIT_EXPERT       = 1u << 1;
static constexpr uint32_t DSG_FLAG_CLEAR_AFTER_FLUSH = 1u << 2;

static constexpr uint32_t DSG_VALID_RUN_FLAGS =
    DSG_FLAG_EMIT_TABLE |
    DSG_FLAG_EMIT_EXPERT |
    DSG_FLAG_CLEAR_AFTER_FLUSH;

static constexpr uint32_t DSG_CAPACITY_USE_SPEC = 0xFFFFFFFFu;

static constexpr uint64_t DSG_FNV_BASIS = 1469598103934665603ull; // 0x14650FB0739D0383
static constexpr uint64_t DSG_FNV_PRIME = 1099511628211ull;
```

**The FNV basis is NOT the canonical FNV offset basis. The exact value above is mandatory.**

### 2.3 Structs

All integer fields are little-endian for FNV folding and raw-byte grading.

```cpp
struct Spec {
    uint32_t magic;        // DSG_SPEC_MAGIC
    uint32_t version;      // DSG_VERSION

    uint32_t table_rows;   // R
    uint32_t dim;          // D, embedding dim and MoE input dim
    uint32_t out_dim;      // O, MoE expert output dim
    uint32_t experts;      // E
    uint32_t top_k;        // K, 1..4
    uint32_t max_tokens;   // max T per run

    uint32_t capacity;     // default per-expert capacity per run
    uint32_t input_dtype;  // DSG_DTYPE_BF16 or DSG_DTYPE_F32

    int32_t  padding_idx;  // if index == padding_idx, table update is skipped before bounds check
    uint32_t flags;        // must be 0 for v1

    uint64_t weight_seed;  // deterministic expert-weight seed

    uint64_t reserved0;    // must be 0
    uint64_t reserved1;    // must be 0
};

struct RunSpec {
    uint32_t magic;             // DSG_RUN_MAGIC
    uint32_t version;           // DSG_VERSION
    uint32_t tokens;            // T, 0..spec.max_tokens
    uint32_t flags;             // DSG_FLAG_*
    uint32_t capacity_override; // DSG_CAPACITY_USE_SPEC or explicit per-expert capacity
    uint32_t reserved0;         // must be 0
    uint64_t user_tag;          // folded into START event, no semantic effect
};

struct OutHeader {
    uint32_t magic;          // DSG_OUT_MAGIC
    uint16_t version;        // DSG_VERSION
    uint16_t header_bytes;   // sizeof(OutHeader)

    uint64_t run_id;         // zero-based id before this run
    uint64_t completed_runs; // run_id + 1 after successful run

    uint32_t tokens;
    uint32_t flags;
    uint32_t effective_capacity;
    uint32_t top_k;

    uint32_t accepted;
    uint32_t dropped_capacity;
    uint32_t dropped_invalid_expert;
    uint32_t skipped_padding;
    uint32_t skipped_invalid_index;
    uint32_t table_segments;
    uint32_t expert_segments;
    uint32_t flush_happened;

    uint64_t total_tokens;
    uint64_t total_accepted;
    uint64_t total_dropped_capacity;
    uint64_t total_dropped_invalid_expert;
    uint64_t total_skipped_padding;
    uint64_t total_skipped_invalid_index;
    uint64_t total_table_segments;
    uint64_t total_expert_segments;
    uint64_t total_flushes;

    uint64_t event_hash;      // per-run canonical event timeline hash
    uint64_t tensor_hash;     // per-run emitted tensor payload hash
    uint64_t state_hash;      // post-run persistent state hash
    uint64_t step_hash;       // FNV over summary fields, defined in §2.14

    uint64_t reserved0;       // 0
    uint64_t reserved1;       // 0
};
```

### 2.4 Shape Family

**Visible public tests:**

| Parameter | Values |
|---|---|
| R (table_rows) | 16, 64, 128, 1024 |
| D (dim) | 1, 7, 8, 16, 32 |
| O (out_dim) | 1, 4, 8, 16 |
| E (experts) | 1, 4, 8, 16 |
| K (top_k) | 1, 2 |
| T (tokens/run) | 0, 1, 8, 31, 64, 257, 512 |
| capacity | 0, 1, 4, 16, 128 |
| dtype | BF16 and F32 |
| run count/state | 1..16 runs per init, with resets between some sequences |

**Hidden tests:**

| Parameter | Values |
|---|---|
| R | 1..8192 |
| D | {1,2,3,5,7,8,16,24,32,48} |
| O | {1,2,4,8,16,24,32} |
| E | 1..64 |
| K | 1..4 |
| T | 0..2048 |
| capacity | 0..4096 |
| dtype | BF16 or F32 |
| runs/init | 1..128 |
| flush pattern | no flush, table-only, expert-only, both, clear-only, emit+clear |

**Hidden generator caps (part of the agent-visible contract):**

```
max_tokens * top_k * dim * out_dim <= 4,194,304   // expert-cell contributions
table_rows * dim                   <= 524,288      // fp32 cells
experts * dim * out_dim            <= 524,288      // fp32 cells
```

### 2.5 Input Layout

`inputs` is a device pointer. Must be 8-byte aligned.

For each run with T = run.tokens, K = spec.top_k, D = spec.dim, O = spec.out_dim, arrays are packed in this exact order, each array start aligned up to 8 bytes:

```cpp
int32_t  embedding_index[T];       // row index per token
int32_t  route_expert[T * K];      // expert id per token-slot
int16_t  route_gate_q15[T * K];    // signed Q15 gate, gate = q / 32768.0
input_t  x[T * D];                 // forward activation entering experts
input_t  dy[T * O];                // upstream expert output gradient
input_t  emb_upstream[T * D];      // direct embedding upstream gradient
```

`input_t` is `uint16_t` raw BF16 bits if `DSG_DTYPE_BF16`, or `uint32_t` raw IEEE fp32 bits if `DSG_DTYPE_F32`.

**Offset rule:**
```cpp
align8(x) = (x + 7) & ~size_t(7)

off_index  = 0
off_expert = align8(off_index  + T * sizeof(int32_t))
off_gate   = align8(off_expert + T * K * sizeof(int32_t))
off_x      = align8(off_gate   + T * K * sizeof(int16_t))
off_dy     = align8(off_x      + T * D * dtype_size)
off_up     = align8(off_dy     + T * O * dtype_size)
input_bytes = align8(off_up + T * D * dtype_size)
```

### 2.6 Output Layout

`outputs` is a device pointer. Must be 8-byte aligned.

```cpp
off_header = 0
off_dx     = sizeof(OutHeader)
off_table  = align8(off_dx + T * D * sizeof(float))             // only if EMIT_TABLE
off_expert = align8(off_table + R * D * sizeof(float))           // only if EMIT_TABLE and EMIT_EXPERT
```

More generally:

- **Payload 1, always present:** `float token_dx[T * D]` in row-major token-major order.
- **Payload 2, iff `run.flags & DSG_FLAG_EMIT_TABLE`:** `float table_grad[R * D]` in row-major order.
- **Payload 3, iff `run.flags & DSG_FLAG_EMIT_EXPERT`:** `float expert_wgrad[E * D * O]` in expert-major, then dim, then out_dim order.

All emitted floats are raw IEEE fp32 bits. NaNs must be canonical `0x7FC00000`.

### 2.7 Deterministic Expert Weights

Expert weights are persistent, deterministic, read-only, generated in `solution_init` from `spec.weight_seed`.

For flattened expert-weight cell:
```cpp
cell = ((e * D + d) * O + o)
```

Generate:
```cpp
uint64_t z = spec.weight_seed
           + 0x9E3779B97F4A7C15ull
           + uint64_t(cell) * 0xBF58476D1CE4E5B9ull;

z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
z = z ^ (z >> 31);

int32_t q = int32_t(z & 0xFFFFu) - 32768;
weight[e,d,o] = round_rn(float(q) * 2^-14);
```

`float(q)` is exact for this range. `2^-14 = 0.00006103515625f` is exact.

### 2.8 Exact Floating-Point Rules

All arithmetic is IEEE-754 fp32, round-to-nearest-even.

**Every input float is canonicalized on load:** if exponent == 255 and mantissa != 0, replace with raw bits `0x7FC00000`. Otherwise preserve raw bits, including ±0 and ±Inf.

**Every fp32 addition or multiplication result is canonicalized immediately:**
```cpp
fmul(a,b) = canonicalize(round_rn(a * b))
fadd(a,b) = canonicalize(round_rn(a + b))
```

**No FMA contraction is allowed.** The expression `a*b + c` must be two primitives:
```cpp
tmp = fmul(a, b)
out = fadd(c, tmp)
```

**Gate conversion:**
```cpp
gate_q15 is int16_t.
gate = round_rn(float(gate_q15) * 2^-15)
2^-15 = 0.000030517578125f.
gate_q15 = -32768 gives exactly -1.0f; 32767 gives 32767/32768.
```

### 2.9 Per-Run Normative Semantics

Let persistent state contain:
```
table_accum[R, D]       fp32, initially +0.0
expert_wgrad[E, D, O]   fp32, initially +0.0
run_id                  uint64_t, initially 0
cumulative counters     uint64_t modulo 2^64
cumulative_event_hash   uint64_t, initially DSG_FNV_BASIS
```

#### Step A: Effective Capacity
```cpp
effective_capacity =
    run.capacity_override == DSG_CAPACITY_USE_SPEC
      ? spec.capacity
      : run.capacity_override;
```

Capacity is per expert and per run. Capacity counters start at zero every run.

#### Step B: Replay MoE Routing in Arrival Order

For token `t = 0..T-1` and slot `k = 0..K-1`:
```cpp
assign = t * K + k
e      = route_expert[assign]
q      = route_gate_q15[assign]
```

**Handling order:**

- If `e < 0 || e >= E`: **invalid expert drop**.  
  Count `dropped_invalid_expert++`. No capacity consumed. No token_dx contribution. No expert_wgrad contribution.

- Else if `capacity_used[e] >= effective_capacity`: **capacity drop**.  
  Count `dropped_capacity++`. No capacity consumed. No token_dx contribution. No expert_wgrad contribution.

- Else **accepted**:  
  `local_pos = capacity_used[e]`. `capacity_used[e]++`. Count `accepted++`. Mark assignment accepted.

For each accepted assignment, update transient `token_dx[t,d]`, initially all `+0.0` for this run:
```cpp
for d in 0..D-1:
    sum = +0.0f
    for o in 0..O-1:
        sum = fadd(sum, fmul(load(dy[t,o]), weight[e,d,o]))
    token_dx[t,d] = fadd(token_dx[t,d], fmul(gate, sum))
```

If a token has multiple accepted slots, slots accumulate in increasing k order.

#### Step C: Build Deterministic Embedding Table Contributions

For each token `t` in arrival order:
```cpp
idx = embedding_index[t]
```

**Handling order:**

- If `idx == spec.padding_idx`: count `skipped_padding++`. No table contribution.
- Else if `idx < 0 || idx >= R`: count `skipped_invalid_index++`. No table contribution.
- Else **valid**: create one table contribution with:
  ```cpp
  row = uint32_t(idx)
  arrival = uint32_t(t)
  sort_key = (uint64_t(row) << 32) | arrival
  token = t
  ```

Sort all table contributions by unsigned `sort_key` ascending (equivalent to: row ascending, token arrival ascending).

For each contiguous segment of equal row, in sorted order:
```cpp
for d in 0..D-1:
    seg_sum = +0.0f
    for contribution j in this row segment, ascending token arrival:
        t = token[j]
        contrib = fadd(load(emb_upstream[t,d]), token_dx[t,d])
        seg_sum = fadd(seg_sum, contrib)

    table_accum[row,d] = fadd(table_accum[row,d], seg_sum)
```

One table segment event is emitted per non-empty row segment.

#### Step D: Build Deterministic Expert Weight-Gradient Contributions

For every accepted assignment `assign = t*K+k`, generate one contribution per expert-weight cell `(e,d,o)`:
```cpp
cell = ((e * D + d) * O + o)
sort_key = (uint64_t(cell) << 32) | uint32_t(assign)
```

Sort all expert-cell contributions by unsigned `sort_key` ascending (equivalent to: expert ascending, d ascending, o ascending, assignment arrival ascending).

For each contiguous segment of equal cell:
```cpp
for contribution j in this cell segment, ascending assignment arrival:
    assign = assignment[j]
    t = assign / K
    k = assign % K
    e = route_expert[assign]
    gate = q15_to_float(route_gate_q15[assign])

    prod = fmul(load(x[t,d]), load(dy[t,o]))
    contrib = fmul(prod, gate)
    seg_sum = fadd(seg_sum, contrib)

expert_wgrad[e,d,o] = fadd(expert_wgrad[e,d,o], seg_sum)
```

One expert segment event is emitted per non-empty expert cell segment.

#### Step E: Flush and Clear

`flush_happened = 1` iff any of these bits are set:
```cpp
DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT | DSG_FLAG_CLEAR_AFTER_FLUSH
```

Output tensor copying happens before clearing.

If `DSG_FLAG_CLEAR_AFTER_FLUSH` is set, then after output payload copy and tensor hash construction:
```cpp
table_accum[:] = +0.0f
expert_wgrad[:] = +0.0f
```

Counters are not cleared by flush. Only `solution_reset` clears counters.

### 2.10 Canonical FNV-1a-64 Primitive

For every folded byte:
```cpp
h = (h ^ byte) * 1099511628211 mod 2^64
```

Initial hash for every independent hash:
```cpp
h = 1469598103934665603  // 0x14650FB0739D0383 -- NOT the canonical FNV basis
```

Integer fields are folded least-significant byte first:
```cpp
fold_u8(x):  one byte x
fold_u16(x): bytes (x >> 0), (x >> 8)
fold_u32(x): bytes (x >> 0), (x >> 8), (x >> 16), (x >> 24)
fold_u64(x): eight bytes, little-endian
fold_i16/i32: convert to unsigned two's-complement width, then fold as u16/u32
fold_f32(x): canonicalize x, reinterpret as uint32_t, fold_u32(bits)
```

### 2.11 Event Timeline Hash

`event_hash` starts at `DSG_FNV_BASIS` every run.

Events are folded in this exact order:

**START event:**
```
tag u8 = 0x01
run_id u64
user_tag u64
T u32
flags u32
effective_capacity u32
K u32
R u32
D u32
O u32
E u32
input_dtype u32
```

**ACCEPT event** — for each accepted assignment, immediately during routing order:
```
tag u8 = 0x02
assign u32
token t u32
slot k u32
expert e i32
gate_q15 i16
local_pos u32
```

**DROP event** — for each dropped assignment, immediately during routing order:
```
tag u8 = 0x03
assign u32
token t u32
slot k u32
expert e i32
gate_q15 i16
reason u8   // 1 = invalid expert, 2 = capacity overflow
```

**PAD-SKIP event** — for each token skipped because `idx == padding_idx`:
```
tag u8 = 0x04
token t u32
index i32
```

**INVALID-INDEX event** — for each token skipped because `idx < 0 || idx >= R` after padding check:
```
tag u8 = 0x05
token t u32
index i32
```

**TABLE-SEGMENT event** — for each non-empty embedding row segment, in sorted row order:
```
tag u8 = 0x10
row u32
sorted_start u32
len u32
first_token_arrival u32
last_token_arrival u32
```

**EXPERT-SEGMENT event** — for each non-empty expert cell segment, in sorted (e,d,o) order:
```
tag u8 = 0x20
expert e u32
dim d u32
out o u32
sorted_start u32
len u32
first_assign_arrival u32
last_assign_arrival u32
```

**FLUSH event** — emitted iff `flush_happened == 1`, after all segment events and before clear:
```
tag u8 = 0x30
flags u32
table_bytes u64   // R * D * 4 if EMIT_TABLE else 0
expert_bytes u64  // E * D * O * 4 if EMIT_EXPERT else 0
clear_after_flush u8
```

### 2.12 Tensor Hash

`tensor_hash` starts at `DSG_FNV_BASIS`.

Fold raw emitted tensor bytes in output payload order:
- `token_dx[T,D]` always
- `table_grad[R,D]` iff `EMIT_TABLE`
- `expert_wgrad[E,D,O]` iff `EMIT_EXPERT`

Header bytes are NOT included in `tensor_hash`.

### 2.13 State Hash

`state_hash` is computed after all run updates, after optional clear, after cumulative counters update, after cumulative event hash update, and after `completed_runs = old_run_id + 1`.

It starts at `DSG_FNV_BASIS` and folds:

```
DSG_SPEC_MAGIC u32
DSG_VERSION u32
R u32
D u32
O u32
E u32
K u32
max_tokens u32
capacity u32
input_dtype u32
padding_idx i32
weight_seed u64

completed_runs u64
total_tokens u64
total_accepted u64
total_dropped_capacity u64
total_dropped_invalid_expert u64
total_skipped_padding u64
total_skipped_invalid_index u64
total_table_segments u64
total_expert_segments u64
total_flushes u64
cumulative_event_hash u64

raw table_accum[R,D] bytes, row-major fp32
raw expert_wgrad[E,D,O] bytes, expert-major fp32
```

All cumulative counters wrap modulo 2^64.

### 2.14 Step Hash

`step_hash` starts at `DSG_FNV_BASIS` and folds:

```
event_hash u64
tensor_hash u64
state_hash u64

run_id u64
completed_runs u64
T u32
flags u32
effective_capacity u32

accepted u32
dropped_capacity u32
dropped_invalid_expert u32
skipped_padding u32
skipped_invalid_index u32
table_segments u32
expert_segments u32
flush_happened u32

total_tokens u64
total_accepted u64
total_dropped_capacity u64
total_dropped_invalid_expert u64
total_skipped_padding u64
total_skipped_invalid_index u64
total_table_segments u64
total_expert_segments u64
total_flushes u64
```

**The grader additionally computes a raw-output FNV over:**
- `OutHeader` raw bytes
- then all emitted payload bytes

**The solver passes only if all header fields, payload bytes, event_hash, tensor_hash, state_hash, step_hash, and cumulative counters match exactly.**

### 2.15 Invalid and OOM Rules

`solution_workspace_bytes(spec)` returns 0 for this v1 contract. All persistent/scratch allocation is internal to `solution_init`.

`solution_init` returns `cudaErrorInvalidValue` and writes `*state = nullptr` if:
- `spec == nullptr` or `state == nullptr`
- magic/version mismatch
- `flags`/reserved fields nonzero
- `input_dtype` not BF16/F32
- `top_k` not in [1,4]
- any of R,D,O,E,max_tokens is zero
- hidden-family cap violated
- any `size_t` multiplication overflows

`solution_run` returns `cudaErrorInvalidValue` and must not mutate state if:
- `state == nullptr`, `run == nullptr`, `inputs == nullptr`, `outputs == nullptr`
- magic/version mismatch
- `reserved0` nonzero
- unknown run flag bit set
- `run.tokens > spec.max_tokens`
- `capacity_override` is not `DSG_CAPACITY_USE_SPEC` and exceeds `max_tokens * top_k`

Invalid experts and invalid embedding indices inside the input are **data events**, not ABI errors.

If allocation fails in `solution_init`, return the CUDA allocation error and free anything already allocated.

---

## 3. Why This Is Strictly Hard

**Duplicate-index gradient scatter is not an atomic-add problem.**  
A mathematically correct `atomicAdd(table[row,d], contrib)` fails because fp32 addition order is nondeterministic and not the specified sorted segment order.

**MoE routing replay and gradient scatter are coupled.**  
`token_dx` depends on accepted expert slots after capacity replay; table gradients depend on `emb_upstream + token_dx`; expert weight gradients depend on accepted assignments only. A solver cannot implement embedding backward independently from MoE routing.

**Capacity overflow changes both numeric outputs and timeline bytes.**  
Per-expert capacity is consumed in token/slot arrival order. Duplicate experts for the same token consume capacity twice. Invalid experts do not consume capacity. These differences affect `token_dx`, expert gradients, drop counters, and event hash.

**Exact fp32 reproduction traps.**  
The contract forbids FMA, requires `fmul` then `fadd`, canonicalizes NaNs after every primitive, and fixes every reduction order. A standard GEMM-style or "equivalent" reduction fails raw-byte checks.

**State persists across runs and flush has two meanings.**  
Emitting accumulated gradients and clearing them are separate flags. State hash is post-clear, tensor hash is pre-clear. Counters persist through flush but reset via `solution_reset`.

**Three independent hashes catch different shortcuts.**  
`tensor_hash` catches emitted bytes, `event_hash` catches routing/segment timeline, and `state_hash` catches persistent hidden state even when no flush emits full tensors.

---
## Clarifications (normative — fold into the contract)
1. **init capacity constraint**: `solution_init` MUST reject (`cudaErrorInvalidValue`) any spec with `capacity > max_tokens * top_k`.
2. **EMIT_EXPERT without EMIT_TABLE**: output offsets advance by the general sequential-offset rule, skipping non-emitted payloads. When EMIT_EXPERT is set and EMIT_TABLE is not, `off_expert = align8(sizeof(OutHeader) + T*D*sizeof(float))`.
3. **raw-output FNV scope**: the output buffer is pre-zeroed before each `solution_run`; the raw-output FNV covers the full used region **including** align8 padding-gap bytes (which are therefore 0).
