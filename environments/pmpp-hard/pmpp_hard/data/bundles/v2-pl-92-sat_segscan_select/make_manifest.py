#!/usr/bin/env python3
"""Build manifest_entry.json for v2-pl-92-sat_segscan_select."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "sat_segscan_select_common.h")) as f:
    common_h = f.read()

prompt = f"""# Task: sat_segscan_select (v2-pl-92)

## 1. Task name + one-line summary

**Task name:** sat_segscan_select

**One-line summary:** Implement an exact SEGMENTED SATURATING inclusive scan
over an int32 stream with packed head-flag bits, producing the dense scan
values, a packed saturation bitmap, the ordered compaction of every
saturated position, and the last value of every segment. The update rule
`p = clamp(p + v, LO, HI)` is NOT associative, segments can span the whole
stream, and the task is perf-gated against a single-pass reference. Every
output is graded exactly -- there are no tolerances anywhere.

## 2. Complete self-contained contract

You implement one `solution.cu` exposing the v2 C ABI declared at the bottom
of the common header:

```cpp
extern "C" size_t solution_workspace_bytes(const SssProblemSpec* spec);
extern "C" cudaError_t solution_init(const SssProblemSpec* spec, void** state_out, cudaStream_t stream);
extern "C" cudaError_t solution_run(void* state, const SssRunSpec* run, const void* inputs, void* outputs, void* workspace, size_t workspace_bytes, cudaStream_t stream);
extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream);
extern "C" void solution_destroy(void* state);
```

`solution_init` may allocate persistent device state sized from
`SssProblemSpec` (max_N). Each `solution_run` receives one `SssRunSpec`
(N, saturation rails lo/hi plus seed/distribution metadata), an `SssInputs`
view (device pointers v[N], flags[ceil(N/32)]), an `SssOutputs` view
(device pointers y, sat_bits, sel_idx, sel_count, seg_last), and the
workspace your `solution_workspace_bytes` requested. It must enqueue all
work on `stream` and return without synchronizing the device.

The ENTIRE normative contract (the sequential semantics that define every
output, the bit-packing conventions, the domain guarantees, and validation
helpers) is in the common header below. It is the single source of truth;
implement it exactly.

### common.h (`sat_segscan_select_common.h`, verbatim)

```cpp
{common_h}```

## 3. Semantics highlights (normative details you must not miss)

- **The scan is NOT associative.** `((a satadd b) satadd c)` differs from
  `(a satadd (b satadd c))` in general; the sequential loop in the header
  is the definition. Any parallel decomposition must reproduce it exactly
  on segments spanning millions of elements -- a single segment covering
  the entire stream occurs, so "parallel across segments, sequential
  within" does not survive the perf gate either.
- **Packed bit streams.** Head flags arrive packed little-endian
  (bit i of word i>>5); flag(0)==1 and padding bits are guaranteed 0. The
  saturation bitmap must be emitted in the SAME packed format, byte-exact
  over the whole buffer, padding bits 0.
- **sat(i) is VALUE equality** (`y[i] == LO || y[i] == HI`), not "the
  clamp fired": exact rail landings without clamping count, and both
  distributions with constant clamping (rails within +-8) and
  distributions that never saturate occur.
- **sel_idx** must list the saturated indices in exactly ascending order;
  only the first sel_count entries are graded (capacity is always N).
- **seg_last[ord]** uses the segment ordinal (flag-count prefix minus 1);
  ends on tile boundaries and single-element segments (all-ones flags) are
  covered. The harness sizes seg_last to exactly S = popcount(flags).
- **Exactness:** with |v| <= 2^20 and rails within +-2^24, all arithmetic
  fits int32 exactly (int64 intermediates are safe); outputs are graded as
  exact integers/bytes and must be deterministic.

## 4. Harness, scenarios, and grading

The test harness runs 54+ seed-driven scenarios: 9 distributions (smooth
never-saturating, tight rails +-8 with constant clamping, rail ping-pong,
engineered exact rail landings and off-by-one probes, one whole-stream
segment, all-ones flags, ragged log-uniform segments, zero-dominated
streams on rails +-1, full-range random) crossed with ragged sizes
(N in [65536, 2^25], nothing divisible by anything). Held-out seeds and
shapes are used for grading. Checks per case: y exact, sat_bits byte-exact,
sel_idx/sel_count exact, seg_last exact, plus output guard sentinels and
input immutability. The grade line is `passed M / M`.

## 5. Rules

- `solution_run` must NOT call cudaMalloc/cudaFree/cudaMallocAsync and must
  not perform any host<->device memcpy (cudaMemsetAsync on device buffers
  is allowed). Kernel launches only; use the provided workspace for
  scratch.
- Inputs are read-only.
- No CUB, Thrust, cuBLAS, cuBLASLt, cuDNN, or CUTLASS headers/libraries.
  Plain CUDA C++ only -- any scan/lookback machinery must be your own
  device code.
- Compilation: `nvcc -O3 -std=c++17 -arch=sm_120`.

## 6. Performance

This task is perf-gated against a single-pass reference on the benchmark
mix (up to 2^25 elements per call, 64-1024 segments). The reference reads
v exactly once: one fused kernel computes the scan with decoupled-lookback
style tile chaining (ticket-ordered tiles, aggregate/prefix states,
windowed lookback) and emits y, the packed saturation bitmap, per-segment
last values and per-tile saturation counts in the same pass, followed by
two tiny kernels for the ordered compaction. The honest
sequential-per-segment baseline is two orders of magnitude slower on the
bench mix. You will need (a) a formulation of the saturating update that
composes associatively, and (b) your own single-pass tile-chaining scan
around it.
"""

entry = {
    "task_id": "v2-pl-92-sat_segscan_select",
    "student_file": "solution.cu",
    "prompt": prompt,
    "gate_type": "perf",
    "test_target": "test_student",
    "bench_target": "bench_student",
    "supported": True,
    "naive_ref_ratio": 114.0,
    "runtime_kind": "cuda",
}

with open(os.path.join(HERE, "manifest_entry.json"), "w") as f:
    json.dump(entry, f, indent=2)
    f.write("\n")

print("prompt_len:", len(prompt))
print("wrote manifest_entry.json")
