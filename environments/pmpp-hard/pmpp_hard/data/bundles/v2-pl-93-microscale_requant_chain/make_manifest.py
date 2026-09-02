#!/usr/bin/env python3
"""Build manifest_entry.json for v2-pl-93-microscale_requant_chain."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "microscale_requant_chain_common.h")) as f:
    common_h = f.read()

prompt = f"""# Task: microscale_requant_chain (v2-pl-93)

## 1. Task name + one-line summary

**Task name:** microscale_requant_chain

**One-line summary:** Implement a deterministic two-stage microscaling
requantization chain: fp32 -> per-32-element E8M0 scales + FP8 E4M3 codes
(bit-level round-to-nearest-even, saturating at 448, subnormals and signed
zeros preserved), then the EXACT stage-1 output values -> per-32-element
msb-based second scales + signed int4 codes through a non-uniform 8-level
LUT with ties to the LARGER index, plus per-row exact int64 squared requant
error, strict stage-1 saturation counts, and per-row two-level word-folded
FNV-1a-64 digests. Every output is graded exactly (byte-exact / exact
integers) -- there are no tolerances anywhere.

## 2. Complete self-contained contract

You implement one `solution.cu` exposing the v2 C ABI declared at the bottom
of the common header:

```cpp
extern "C" size_t solution_workspace_bytes(const MrqProblemSpec* spec);
extern "C" cudaError_t solution_init(const MrqProblemSpec* spec, void** state_out, cudaStream_t stream);
extern "C" cudaError_t solution_run(void* state, const MrqRunSpec* run, const void* inputs, void* outputs, void* workspace, size_t workspace_bytes, cudaStream_t stream);
extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream);
extern "C" void solution_destroy(void* state);
```

`solution_init` may allocate persistent device state sized from
`MrqProblemSpec` (max_R, max_C). Each `solution_run` receives one
`MrqRunSpec` (R, C plus seed/distribution metadata), an `MrqInputs` view
(device pointer x[R*C]), an `MrqOutputs` view (device pointers e4m3_codes,
q4_packed, sf1, sf2, row_err, row_digest, sat1_count), and the workspace
your `solution_workspace_bytes` requested. It must enqueue all work on
`stream` and return without synchronizing the device.

The ENTIRE normative contract (the E4M3 grid, every rounding and tie rule
of both stages, the sentinel scale paths, the packing conventions, the
fixed-point error definition, the digest definition, and validation
helpers) is in the common header below. It is the single source of truth;
implement it exactly.

### common.h (`microscale_requant_chain_common.h`, verbatim)

```cpp
{common_h}```

## 3. Semantics highlights (normative details you must not miss)

- **Every rounding decision is exactly decidable in integers.** The
  recommended implementation decomposes |x| into (mantissa, exponent) and
  rounds on the E4M3 grid with integer shift/compare logic. Dividing by
  the scale in fp32 and rounding the result double-rounds on the DENORM
  and midpoint distributions.
- **Stage-1 RNE:** even-significand ties, including across binade
  boundaries (15.5 steps -> carry into the next binade) and at the
  subnormal/normal boundary; u == 2^-10 exactly rounds to ZERO; the NaN
  code (E=15,m=7) is never produced -- everything beyond 448 saturates to
  (E=15,m=6). sat1_count counts STRICTLY u > 448 (u == 448 exactly does
  not count, and saturating rounds like u in (448, 464) do count).
- **sexp1 = max(floor_log2(amax) - 8, -127)** from the fp32 bit pattern
  (subnormal amax uses the mantissa msb path). Zero-input blocks: sf1 =
  0x00, sign-only codes (the sign of -0.0 IS preserved in the E4M3 byte).
- **Stage 2 operates on the stage-1 OUTPUT**, as integer magnitudes
  M = value * 2^9. qmax's msb is the second scale (sf2 = msb); blocks
  whose nonzero inputs all quantize to ZERO codes take the sf2 = 0xFF
  sentinel path. The LUT index is nearest-with-ties-to-LARGER-index --
  the opposite tie direction from stage 1 -- and equals the count of the
  seven exact integer boundary tests 16*M >= p << msb,
  p in {{1,3,5,8,13,20,28}}. Every LUT midpoint is E4M3-representable, so
  exact stage-2 ties occur constantly.
- **Packing:** q4 packs two codes per byte (low nibble = even element);
  for odd C the final high nibble is ZERO, sign bit included.
- **row_err:** fixed point at 2^-12 in the scaled domain:
  sum of (8*M - p_level << msb)^2 as exact int64 (p_level in
  {{0,1,2,3,5,8,12,16}}), zero blocks contribute 0.
- **row_digest:** non-canonical FNV-1a-64 (basis 1469598103934665603,
  prime 1099511628211) folding 64-bit little-endian WORDS of the byte
  stream [e4m3 row | q4 row | sf1 row | sf2 row] (zero-padded final word);
  word w belongs to substream w mod 8; combine folds sub[0..7] in order.

## 4. Harness, scenarios, and grading

The test harness runs 54+ seed-driven scenarios: 9 distributions (smooth,
stage-1 grid midpoints in both the normal and subnormal regions, near-448
saturation probes including exactly 448 and 464, stage-2 LUT-midpoint
landings plus off-by-one neighbours, all-zero blocks with -0.0 signs,
subnormal-amax blocks that quantize to all-zero codes, +-0.0 mixtures,
power-of-two binade edges +-1 ulp, random finite bit patterns) crossed
with ragged shapes (R in [128, 8192], C in [64, 4096], nothing divisible
by anything). Held-out seeds and shapes are used for grading. Checks per
case: e4m3_codes / q4_packed / sf1 / sf2 byte-exact, row_err / row_digest /
sat1_count exact, plus output guard sentinels and input immutability. The
grade line is `passed M / M`.

## 5. Rules

- `solution_run` must NOT call cudaMalloc/cudaFree/cudaMallocAsync and must
  not perform any host<->device memcpy (cudaMemsetAsync on device buffers
  is allowed). Kernel launches only; use the provided workspace for
  scratch.
- Inputs are read-only.
- No CUB, Thrust, cuBLAS, cuBLASLt, cuDNN, or CUTLASS headers/libraries.
  Plain CUDA C++ only.
- Compilation: `nvcc -O3 -std=c++17 -arch=sm_120`.

## 6. Performance

This task is perf-gated against a fully fused reference on the benchmark
mix (up to 8192x4096, ~86M elements per call). The reference reads x
exactly once and materializes nothing intermediate in global memory: one
kernel chains both block reductions, the integer E4M3 rounder, the LUT
indexer, the error/saturation accumulators, the nibble packing, and the
per-row digest chains from a shared-memory staging of each row's output
stream. Multi-pass implementations that materialize the E4M3 magnitude
matrix or the nibble matrix are many times slower than the gate. Aim for
one pass: warp-per-row block loops with both reductions in-warp
(__reduce_max_sync twice), pure-integer rounding, shared staging laid out
like the digest stream, and parallel per-row digest word chains.
"""

entry = {
    "task_id": "v2-pl-93-microscale_requant_chain",
    "student_file": "solution.cu",
    "prompt": prompt,
    "gate_type": "perf",
    "test_target": "test_student",
    "bench_target": "bench_student",
    "supported": True,
    "naive_ref_ratio": 28.0,
    "runtime_kind": "cuda",
}

with open(os.path.join(HERE, "manifest_entry.json"), "w") as f:
    json.dump(entry, f, indent=2)
    f.write("\n")

print("prompt_len:", len(prompt))
print("wrote manifest_entry.json")
