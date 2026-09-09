# PMPP Hard

PMPP Hard is a 69-task environment for agents that write and optimize GPU kernels. The release pool contains 60 CUDA tasks, 8 Triton tasks, and 1 CUTLASS task from [SinatrasC/pmpp-hard](https://github.com/SinatrasC/pmpp-hard), release `v0.1.1`.

Submissions must compile, produce the expected output, and satisfy the configured performance gate on supported tasks. The release targets NVIDIA Blackwell: `sm_120` on an RTX PRO 6000 or RTX 5090, with `sm_100` B200 support.

## Trust boundary

The agent receives the task contract, solution stub, allowlisted build helpers, and an optional non-authoritative sanity grader. It does not receive reference implementations or authoritative grader files, and its sandbox has framework-only network access.

After the rollout, only the submitted source is copied into a new GPU scoring sandbox. That sandbox performs authoritative compilation, correctness checks, and five interleaved student/reference measurements. Performance-gated tasks pass when the median student/reference runtime ratio meets the task-specific target (default `1.25`) and output digests match. KernelGuard supplies the additional source and timing-policy gates.

For 54 CUDA tasks, one CUTLASS task, and all eight Triton tasks, the scorer draws an independent 64-bit input parameter for each task and rollout, after submission. CUDA/CUTLASS graders use a fresh odd PRNG/input constant or a task-specific scenario transformation; Triton graders use a fresh seed with their existing per-case offsets. Oracle logic, problem-size limits and pass criteria stay unchanged. Task 44 expands one operation batch to exercise its existing flush/compaction coverage requirements explicitly; the other transformations preserve shapes. Public sanity graders remain deterministic for agent debugging. The completed trace records the parameter under `info.pmpp.correctness_inputs` for reproduction, and the lever vector identifies policy `independent-correctness-inputs-v2`; scores should not be treated as identical to the original fixed-input release.

Task 09 (`penalty_filter_sample`) uses an independently drawn, fixed 64-bit sanity constant validated against its reference on all seven cases. Its scoring inputs remain fixed because fresh draws expose disagreement between the GPU reference and CPU oracle on exact sampled-token assertions. Its probability and token checks remain unchanged.

Tasks 13, 37, 45, 51, 63, and 66 have no public sanity grader but support the same correctness transformation. Each passed five independent input draws against its release reference on an `sm_120` GPU, including task 13 in the CUTLASS image. Additional scenario transformations are bound to exact release-source anchors and fail closed if those anchors change:

| Tasks      | Scoring-only transformation                                                                                                                                                                                                                                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 30         | Bijective vertex relabeling, with CSR rows, edge-update indices, source and reset vertices remapped together. Graph connectivity, weights, shapes and update schedules are preserved.                                                                                                                                                                                          |
| 36, 52     | Bijective nonnegative token/packet-ID relabeling. Shared prefixes, repeated IDs and negative invalid IDs are preserved.                                                                                                                                                                                                                                                        |
| 48, 49, 50 | Fresh bounded payload magnitudes, preserving signs and zero cases. Keys, operation sequences and capacity limits are unchanged.                                                                                                                                                                                                                                                |
| 53         | Fresh positive allocation sizes; zero/negative cases, object links, finalizer settings and GC schedules are unchanged.                                                                                                                                                                                                                                                         |
| 61, 64, 68 | Fresh producer/consumer, worker-task and collective payload seeds, respectively. Dependencies, counters, masks and schedules are unchanged.                                                                                                                                                                                                                                    |
| 67         | Fresh enqueue payload seeds only. The released schedules remain fixed: randomizing the schedule PRNG previously caused reference illegal memory accesses for parameters `0x1575a308485d39fd` and `0x1f9653e49c0673b9`. This does not fix that broader reference issue.                                                                                                         |
| 44         | Fresh PRNG/input constant plus explicit empty compaction, empty flush and full-level flush operations in the invalid-operation scenario. These guarantee `compact_empty`, `flush_empty` and `flush_oom` coverage, which some draws previously missed. The last condition fills the four available L0 file slots before attempting a fifth flush. No coverage check is removed. |

Per-rollout input variation covers 63 of 69 tasks. Tasks 09 and 70–74 remain fixed: task 09 needs its reference/oracle sampling disagreement resolved, while 70–74 compare against embedded golden checksums tied to their released inputs. Enabling those five requires seed-aware expected-output generation, not simply changing a constant. Their fixed public cases retain a precomputation risk.

Coverage counts tasks with varying inputs, not randomized shapes or wholly new scenario schedules. Payload/label transformations retain the original structural cases, and bounded fields do not have 64 effective random bits each. Even randomized inputs do not make the scorer a complete security boundary: submitted code runs beside its grader, and KernelGuard/source checks are best-effort defenses against grader inspection or tampering. This benchmark does not claim immunity to reward hacking under adversarial RL pressure. Hidden shapes are not enabled.

### Performance inputs and complete timing pairs

All 48 active performance benchmarks already consume `PMPP_BENCH_SEED`: 40 CUDA tasks and one CUTLASS task use `pmpp::bench_seed()`, and seven Triton tasks derive their input seeds from the environment. The fixed PRNG constants in public source are mixing constants, not an unchanging scoring input stream. Task 33's fixed Triton benchmark is not performance-gated.

Policy `paired-inputs-u64-complete-v2` draws an independent unsigned 64-bit seed after submission, shared by the student and reference across all five pairs in that comparison. The existing input generators, shapes, oracle checks, and timing loops are retained. Private copies of benchmarks 18 and 26 wrap Torch seed-offset additions modulo `2**64`; public bundles and sanity inputs are unchanged. The other Triton generators retain their existing 62- or 63-bit masks, so a 64-bit draw is not a claim of 64 effective bits in every generated input.

Each requested pair must finish with two zero exit codes, exactly one finite positive timing per process, and matching 64-bit output digests. Missing, duplicated, malformed, timed-out, or mismatched samples cannot be discarded to accept a shorter comparison. A failing reference is an infrastructure error; a failing student is a solution failure. `info.perf_ratio` records the seed (hexadecimal), policy, requested/completed pair counts, process exit codes, output digests, log tails, and unrounded timing samples. Seed values belong to completed private traces, not agent feedback.

Fresh input values can change measured ratios even on the same GPU; no invariance to seed or GPU load is assumed. Existing difficulty labels and historical scores are not recalibrated by this change. Compare results using the complete lever vector, and calibrate on an otherwise idle GPU before publishing new performance claims. The in-process GPU lock serializes graders, not agent-side self-checks or other host GPU workloads. Input variation and matching digests are regression defenses, not proof of honest computation or protection against arbitrary grader tampering.

Task 30's provisioned benchmark also normalizes a zero external-workspace request to a one-byte allocation, matching its correctness grader. Its contract permits persistent storage in `solution_init`; returning zero external workspace must not itself cause a performance failure. This correction is bound to the release source guard, is applied to both agent and scoring copies, and does not modify the submitted kernel or public bundle. Nonzero workspace requests, input generation and timing loops are unchanged. Other tasks' workspace contracts are not changed by this targeted correction.

### GPU execution and fast-kernel diagnostics

Triton submissions must define a JIT kernel and have a matching kernel observed by the runtime GPU probe, including on performance-gated tasks. Valid compiled or cached launchers need not spell the call as `kernel[grid](...)`. Defining an unused kernel or using only Torch does not satisfy the runtime requirement. Private residency metadata retains the observed kernel names and `executed_jit_names` matching the submission. Performance timing itself is not profiled.

For correctness-only tasks, actual GPU activity remains required. The reference-relative 2% GPU-time floor is now diagnostic by default: a sufficiently fast GPU algorithm can honestly fall below it. Low-time observations are retained under `info.pmpp.residency.time_warning`; they no longer alone invalidate a correct submission. Set `residency_time_mode="enforce"` to restore the old time-floor rejection behavior, or `residency_gpu_frac=0` to skip that optional baseline. Missing or ambiguous student probe data remains an infrastructure error. Policy `runtime-kernels-v2`, the time mode, threshold and launch requirement are recorded in the lever vector. Kernel activity alone is not full computation attribution; diagnostic mode can admit decoy work that the remaining checks miss, and suspicious submissions still require review.

Saved submissions can be regraded with the support scripts below, but such diagnostics must remain separate from their original model rollouts and scores.

## Runtime images

The agent and clean scorer use task-specific CUDA 12.8 images. Build the images from the repository root:

```bash
scripts/pmpp-hard/build-images.sh
```

This produces:

- `pmpp-cuda-agent:cu128`
- `pmpp-triton:cu128`
- `pmpp-cutlass:12.8.1`

Docker must have NVIDIA Container Toolkit access to a compatible GPU. The images pre-cache the dependencies required to start the configured harness after network access is blocked. The published configs use local Docker images; remote providers require equivalent images in an accessible registry.

Triton images must already contain working `torch` and `triton` imports. Agent setup and scoring probe these dependencies and fail immediately with an image-content error when they are missing; neither sandbox attempts package installation at runtime.

## Evaluation

Install the environment from the repository root:

```bash
uv pip install -e ./environments/pmpp-hard
```

Validate the one-task configuration without running a model:

```bash
uv run eval @ configs/pmpp-hard/smoke.toml --dry-run
```

Then run the smoke or full task matrix:

```bash
uv run eval @ configs/pmpp-hard/smoke.toml
uv run eval @ configs/pmpp-hard/full.toml
```

The smoke configuration selects `v2-pl-34-fused_csr_spmm_topk`. The full configuration enumerates all 69 release tasks and resolves each task to its CUDA, Triton, or CUTLASS image.

Local reference/input checks, stale-output controls, and saved-submission replay commands are documented in [`scripts/pmpp-hard/README.md`](../../scripts/pmpp-hard/README.md). These do not call a model or publish results.
