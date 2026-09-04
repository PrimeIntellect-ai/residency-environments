# PMPP Hard

PMPP Hard is a 69-task environment for agents that write and optimize GPU kernels. The release pool contains 60 CUDA tasks, 8 Triton tasks, and 1 CUTLASS task from [SinatrasC/pmpp-hard](https://github.com/SinatrasC/pmpp-hard), release `v0.1.1`.

Submissions must compile, produce the expected output, and satisfy the configured performance gate on supported tasks. The release targets NVIDIA Blackwell: `sm_120` on an RTX PRO 6000 or RTX 5090, with `sm_100` B200 support.

## Trust boundary

The agent receives the task contract, solution stub, allowlisted build helpers, and an optional non-authoritative sanity grader. It does not receive reference implementations or authoritative grader files, and its sandbox has framework-only network access.

After the rollout, only the submitted source is copied into a new GPU scoring sandbox. That sandbox performs authoritative compilation, correctness checks, and five interleaved student/reference measurements. Performance-gated tasks pass when the median student/reference runtime ratio is at most `1.25` and output digests match. KernelGuard supplies the additional source and timing-policy gates.

For 37 CUDA tasks with a one-byte sanity PRNG perturbation and all eight Triton tasks, the scorer now draws an independent 64-bit input parameter for each task and rollout, after submission. CUDA graders use a fresh odd PRNG/input constant; Triton graders use a fresh seed with their existing per-case offsets. The test shapes, oracle logic, and pass criteria stay unchanged. Public sanity graders remain deterministic for agent debugging. The completed trace records the parameter under `info.pmpp.correctness_inputs` for reproduction, and the lever vector identifies this scoring policy; scores should not be treated as identical to the original fixed-input release.

Task 09 (`penalty_filter_sample`) uses an independently drawn, fixed 64-bit sanity constant validated against its reference on all seven cases. Its scoring inputs remain fixed because fresh draws expose disagreement between the GPU reference and CPU oracle on exact sampled-token assertions. Its probability and token checks remain unchanged.

Per-rollout randomization covers 45 of 69 tasks. The other 24 retain their release correctness cases: task 09, five sanity graders (tasks 70–74) bound to embedded golden checksums, and 18 tasks without a public sanity grader. Those fixed public cases retain a precomputation risk. Even randomized inputs do not make the scorer a complete security boundary: submitted code runs beside its grader, and KernelGuard/source checks are best-effort defenses against grader inspection or tampering. This benchmark does not claim immunity to reward hacking under adversarial RL pressure. Hidden shapes are not enabled.

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
