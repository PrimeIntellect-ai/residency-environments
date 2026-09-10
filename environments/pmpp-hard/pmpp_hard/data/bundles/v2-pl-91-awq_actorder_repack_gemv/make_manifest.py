#!/usr/bin/env python3
"""Build manifest_entry.json for v2-pl-91-awq_actorder_repack_gemv."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "awq_actorder_repack_gemv_common.h")) as f:
    common_h = f.read()

prompt = f"""# Task: awq_actorder_repack_gemv (v2-pl-91)

## 1. Task name + one-line summary

**Task name:** awq_actorder_repack_gemv

**One-line summary:** Implement an exact AWQ act-order repack + dequant-GEMV
pipeline: AWQ-packed int4 weights (8 output channels per u32, AWQ weight
nibble order) with grouped int4 zero points (AWQ zero order) and fp32 scales
are repacked along a STABLE group-sorted permutation of the input channels
(GPTQ act-order style; the permutation itself is a graded output) into
marlin-flavoured 16x64 swizzled tiles, fused with a deterministically ordered
fp32 dequant-dot per output column, exact int32 column sums, and per-column
two-level word-folded FNV-1a-64 digests. Every output is graded exactly
(byte-exact / bit-exact / exact integers) -- there are no tolerances anywhere.

## 2. Complete self-contained contract

You implement one `solution.cu` exposing the v2 C ABI declared at the bottom
of the common header:

```cpp
extern "C" size_t solution_workspace_bytes(const AwqProblemSpec* spec);
extern "C" cudaError_t solution_init(const AwqProblemSpec* spec, void** state_out, cudaStream_t stream);
extern "C" cudaError_t solution_run(void* state, const AwqRunSpec* run, const void* inputs, void* outputs, void* workspace, size_t workspace_bytes, cudaStream_t stream);
extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream);
extern "C" void solution_destroy(void* state);
```

`solution_init` may allocate persistent device state sized from
`AwqProblemSpec` (max_K, max_N, max_G). Each `solution_run` receives one
`AwqRunSpec` (K, N, G plus seed/distribution metadata), an `AwqInputs` view
(device pointers qweight, qzeros, scales, g_idx, x), an `AwqOutputs` view
(device pointers rq_atoms, col_dot, col_digest, col_zsum, perm), and the
workspace your `solution_workspace_bytes` requested. It must enqueue all work
on `stream` and return without synchronizing the device.

The ENTIRE normative contract (both AWQ nibble orders, the stable rank
formula, the tile/swizzle layout, the pinned reduction order, the digest
definition, the domain guarantee, and validation helpers) is in the common
header below. It is the single source of truth; implement it exactly.

### common.h (`awq_actorder_repack_gemv_common.h`, verbatim)

```cpp
{common_h}```

## 3. Semantics highlights (normative details you must not miss)

- **perm is graded.** It is the stable ascending argsort of `g_idx` (group
  ascending, ties by original channel index) and must be produced on device.
  Everything else -- repack bytes, dot accumulation order, digest stream --
  is defined over PERMUTED positions `j` with source channel `perm[j]`.
- **Three distinct nibble layouts.** Source weights: 8 output channels per
  word at physical lanes wlane(t) = 0,2,4,6,1,3,5,7. Zero points: zlane(t) =
  0,4,1,5,2,6,3,7. Destination: 8 PERMUTED input channels per word (wlane
  along j) with the h/i row split and the ((nt&7)<<3)|(nt>>3) column swizzle.
  Mixing any two of them passes no hidden case.
- **Padding:** every word of rq_atoms (128*TA*TN u32) must be written;
  out-of-range nibbles and fully-padding words are 0. The harness pre-fills
  outputs with a sentinel, so lazy padding fails. Ragged K (vs 16-row tiles)
  and ragged N (vs 8-lane words and 64-column atoms) both occur.
- **col_dot:** 16 pinned lanes over permuted positions;
  `partial[l] = fadd_rn(partial[l], fmul_rn(fmul_rn((float)qz, s), x))` in
  ascending j, then the fixed stride 8,4,2,1 tree. IEEE fp32
  round-to-nearest-even, NO FMA contraction, NO FTZ: subnormal scales,
  negative scales, and +-0.0 scales/x all occur and are graded bit-exact.
  The harness guarantees no product or partial sum can overflow.
- **col_digest:** non-canonical FNV-1a-64 (basis 1469598103934665603, prime
  1099511628211) folding 64-bit LITTLE-ENDIAN WORDS of the byte stream
  [K two's-complement qz bytes in permuted order][4G scale bytes LE]. Word w
  belongs to substream w mod 8; when K % 8 != 0 one straddle word mixes qz
  and scale bytes; the final word is zero-padded. Combine folds sub[0..7] in
  order. ALL G groups' scale bytes appear, including empty groups.
- **col_zsum:** plain int32 sum of qz over all positions (order-free).

## 4. Harness, scenarios, and grading

The test harness runs 54+ seed-driven scenarios: 9 distributions (balanced
groups, reversed block maps, heavy skew with empty groups, all-equal g_idx,
q == z everywhere with mixed-sign and +-0.0 scales, ~50% zero scales,
subnormal scales, exact power-of-two scales/x with 0/15 nibble runs, capped
random bit patterns) crossed with ragged shapes (K in [256, 8192], N in
[64, 4096], G in [4, 512], nothing divisible by anything). Held-out seeds
and shapes are used for grading. Checks per case: perm exact, rq_atoms
byte-exact, col_dot bit-exact, col_digest exact, col_zsum exact, plus output
guard sentinels and immutability of all five input buffers. The grade line
is `passed M / M`.

## 5. Rules

- `solution_run` must NOT call cudaMalloc/cudaFree/cudaMallocAsync and must
  not perform any host<->device memcpy (cudaMemsetAsync on device buffers is
  allowed). Kernel launches only; use the provided workspace for scratch.
- Inputs are read-only.
- No CUB, Thrust, cuBLAS, cuBLASLt, cuDNN, or CUTLASS headers/libraries.
  Plain CUDA C++ only (the stable rank must be your own device code).
- Compilation: `nvcc -O3 -std=c++17 -arch=sm_120` (defaults: no fast-math,
  no FTZ, FMA contraction allowed by the compiler but the pinned reduction
  is defined WITHOUT contraction -- use __fmul_rn/__fadd_rn where it
  matters).

## 6. Performance

This task is perf-gated against a highly fused reference on the benchmark
mix (K up to 8192, N up to 4096, G up to 512, ~33M weights per call). The
reference computes the stable rank with a counting sort, transposes the
scales once into workspace, then does everything else in ONE software-
pipelined kernel: it reads qweight exactly once, never materializes the
unpacked q matrix in global memory, and feeds the repacked tiles, both
pinned dot chains per thread, the column sums, and all eight per-column
digest word-chains from the same shared-memory staging of each 64-row
permuted chunk. Multi-pass implementations that materialize unpacked or
dequantized matrices are several times slower than the gate. Aim for a
single fused pass over the permuted rows with prefetching (the act-order
gather is latency-bound), plus a scale layout that keeps the per-chain
scale lookups cache-resident.
"""

entry = {
    "task_id": "v2-pl-91-awq_actorder_repack_gemv",
    "student_file": "solution.cu",
    "prompt": prompt,
    "gate_type": "perf",
    "test_target": "test_student",
    "bench_target": "bench_student",
    "supported": True,
    "naive_ref_ratio": 5.7,
    "runtime_kind": "cuda",
}

with open(os.path.join(HERE, "manifest_entry.json"), "w") as f:
    json.dump(entry, f, indent=2)
    f.write("\n")

print("prompt_len:", len(prompt))
print("wrote manifest_entry.json")
