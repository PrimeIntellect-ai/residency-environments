#!/usr/bin/env python3
"""Build manifest_entry.json for v2-pl-90-mxfp4_atom_quant_dot."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "mxfp4_atom_quant_dot_common.h")) as f:
    common_h = f.read()

prompt = f"""# Task: mxfp4_atom_quant_dot (v2-pl-90)

## 1. Task name + one-line summary

**Task name:** mxfp4_atom_quant_dot

**One-line summary:** Implement an exact MXFP4 quantization pipeline: fp32
matrix -> per-32-element E8M0 scales + FP4 E2M1 codes (round-to-nearest,
ties-to-even-code, saturating), packed into CUTLASS-style swizzled 128x64
payload atoms and 128x128 scale-factor atoms, with a fused deterministically
ordered fp32 dequant-dot epilogue, per-row saturation counts, and per-row
two-level FNV-1a-64 digests. Every output is graded exactly (byte-exact /
bit-exact / exact integers) -- there are no tolerances anywhere.

## 2. Complete self-contained contract

You implement one `solution.cu` exposing the v2 C ABI declared at the bottom
of the common header:

```cpp
extern "C" size_t solution_workspace_bytes(const MxqProblemSpec* spec);
extern "C" cudaError_t solution_init(const MxqProblemSpec* spec, void** state_out, cudaStream_t stream);
extern "C" cudaError_t solution_run(void* state, const MxqRunSpec* run, const void* inputs, void* outputs, void* workspace, size_t workspace_bytes, cudaStream_t stream);
extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream);
extern "C" void solution_destroy(void* state);
```

`solution_init` may allocate persistent device state sized from
`MxqProblemSpec` (max_R, max_C). Each `solution_run` receives one
`MxqRunSpec` (R, C plus seed/distribution metadata), an `MxqInputs` view
(device pointers x[R*C], v[C]), an `MxqOutputs` view (device pointers
pay_atoms, sf_atoms, row_dot, row_digest, sat_count), and the workspace your
`solution_workspace_bytes` requested. It must enqueue all work on `stream`
and return without synchronizing the device.

The ENTIRE normative contract (every rounding rule, layout formula, reduction
order, digest definition, domain guarantee, and validation helper) is in the
common header below. It is the single source of truth; implement it exactly.

### common.h (`mxfp4_atom_quant_dot_common.h`, verbatim)

```cpp
{common_h}```

## 3. Semantics highlights (normative details you must not miss)

- **E8M0 scale:** `e = floor_log2(amax)` is defined on the fp32 BIT PATTERN,
  including the subnormal path (`msb_index(M) - 149`); `sexp = max(e-2,-127)`;
  `sf_byte = sexp + 127`. All-zero blocks (only +-0.0) write sf_byte 0x00 and
  sign-only codes: the sign of `-0.0` IS preserved in the E2M1 nibble.
- **E2M1 rounding:** the threshold table with `<=` / `<` at each midpoint is
  the tie rule (ties to EVEN magnitude code), saturating at code 7. Each
  `T(t) = t * 2^sexp` is exactly representable; comparisons are exact.
- **sat_count[r]** counts strictly `|x| > 6 * 2^sexp` lanes (not `code == 7`).
- **Two nibble pairings:** the ATOM byte pairs elements (u, u+32); the LINEAR
  logical stream (used only inside the digest) pairs (2j, 2j+1) with a zero
  high nibble when C is odd. Do not mix them up.
- **Padding:** EVERY byte of pay_atoms (4096*AM*AK) and sf_atoms (512*AM*AS)
  must be written; padding bytes are 0x00. The harness pre-fills output
  buffers with a sentinel, so unwritten padding fails.
- **row_dot:** 128 pinned partial lanes, `partial[l]` accumulates
  `fadd_rn(partial, fmul_rn(dq, v[k]))` over `k = l, l+128, ...` in ascending
  k, then the fixed stride 64,32,...,1 tree. fp32, round-to-nearest-even, no
  FMA contraction, no FTZ. Graded bit-exact as u32 patterns. The harness
  guarantees no product or partial sum can overflow to infinity.
- **row_digest:** non-canonical FNV-1a-64 (basis 1469598103934665603,
  prime 1099511628211), fold = `(h ^ byte) * prime` mod 2^64. Stream = Kb
  logical payload bytes then S scale bytes; split into 8 equal ranges
  (`sub_len = ceil((Kb+S)/8)`, last ranges may be empty); row digest = fold of
  the 8 sub-digests, each folded as 8 bytes little-endian (low byte first).

## 4. Harness, scenarios, and grading

The test harness runs 54+ seed-driven scenarios: 9 distributions (smooth,
exact tie midpoints, saturation-heavy, all-zero blocks, denormal-amax blocks,
+-0.0 mixtures, power-of-two binade edges, subnormal clamp-low, random finite
bit patterns with exponent field <= 0xF0) crossed with ragged shapes
(R in [128, 16384], C in [64, 6144], nothing divisible by anything). Held-out
seeds and shapes are used for grading. Checks per case: pay_atoms byte-exact,
sf_atoms byte-exact, row_dot bit-exact, row_digest exact, sat_count exact,
plus output guard sentinels and input immutability. The grade line is
`passed M / M`.

## 5. Rules

- `solution_run` must NOT call cudaMalloc/cudaFree/cudaMallocAsync and must
  not perform any host<->device memcpy (cudaMemsetAsync on device buffers is
  allowed). Kernel launches only; use the provided workspace for scratch.
- Inputs are read-only.
- No CUB, Thrust, cuBLAS, cuBLASLt, cuDNN, or CUTLASS headers/libraries.
  Plain CUDA C++ only.
- Compilation: `nvcc -O3 -std=c++17 -arch=sm_120` (defaults: no fast-math, no
  FTZ, FMA contraction allowed by the compiler but the pinned reduction is
  defined WITHOUT contraction -- use __fmul_rn/__fadd_rn where it matters).

## 6. Performance

This task is perf-gated against a highly fused reference on the benchmark
shape mix (R up to 16384, C up to 6143, ~94M elements per call). The
reference reads x exactly once and materializes neither the code matrix nor
the dequantized matrix in global memory; multi-pass implementations that
materialize intermediates are several times slower than the gate. Aim for a
single fused pass: block-per-rows quantization with in-register dot partials
(the 128 pinned partials map exactly onto 4 accumulators per lane for a
32-lane warp), shared-memory staging for packed bytes, wide vectorized atom
stores, and parallel per-row digest chains.
"""

entry = {
    "task_id": "v2-pl-90-mxfp4_atom_quant_dot",
    "student_file": "solution.cu",
    "prompt": prompt,
    "gate_type": "perf",
    "test_target": "test_student",
    "bench_target": "bench_student",
    "supported": True,
    "naive_ref_ratio": 6.1,
    "runtime_kind": "cuda",
}

with open(os.path.join(HERE, "manifest_entry.json"), "w") as f:
    json.dump(entry, f, indent=2)
    f.write("\n")

print("prompt_len:", len(prompt))
print("wrote manifest_entry.json")
