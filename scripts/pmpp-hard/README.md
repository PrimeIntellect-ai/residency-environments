# PMPP Hard runtime images

Build the three local GPU images from the repository root:

```bash
scripts/pmpp-hard/build-images.sh
```

The CUDA agent image supplies CUDA 12.8, Python, and the utilities needed by a coding harness. All three images pin `uv` and pre-cache the dependencies used to launch the configured Verifiers harness with agent network access disabled. Triton tasks use a CUDA 12.8 image with pinned PyTorch and Triton versions. The CUTLASS image includes the pinned CUTLASS headers used by the release task.

All images require an NVIDIA Container Toolkit installation and a compatible host driver. The image definitions pin their base image and specialized toolchain inputs. Publish equivalent images to an accessible registry before using a remote GPU runtime.

To check all 63 randomized correctness graders and task 09's independently reseeded sanity grader against their trusted release references, run:

```bash
uv run scripts/pmpp-hard/validate-inputs.py --output /tmp/pmpp-input-validation
```

This runs two independent input draws per randomized task and two checks of the fixed task 09 sanity cases on local GPU Docker, with external network access blocked. CUDA, CUTLASS, and Triton tasks use their corresponding images; override them with `--cuda-image`, `--cutlass-image`, and `--triton-image`. It writes per-run logs, image tags, and a JSON result manifest to an empty output directory and verifies that every selected task and rollout completed. Unknown or unsupported task selections are rejected. It checks correctness without model calls or performance measurements; full evaluation still uses the configs under `configs/pmpp-hard/`.

To repeat the six newly enabled tasks with five independent draws each:

```bash
uv run scripts/pmpp-hard/validate-inputs.py --rollouts 5 --output /tmp/pmpp-extra-input-validation \
  v2-pl-13-fused_gemm_nvfp4_epilogue \
  v2-pl-37-deterministic_mergeable_quantile_sketch \
  v2-pl-45-chunked_prefill_scheduler \
  v2-pl-51-mesi_directory \
  v2-pl-63-mk_event_tensor_runtime \
  v2-pl-66-mk_schedule_planner
```

Tasks 30, 36, 44, 48, 49, 50, 52, 53, 61, 64, 67 and 68 have additional scoring-only transformations described in the environment README. Validate them with the same command and positional IDs. Task 44 now explicitly covers empty compaction, empty flushing and flushing a full level; task 67 varies payloads while preserving the released schedules that avoid its broader reference failure.

For guarded CUDA/CUTLASS outputs, check reproducibility and actual output variation with an A/A/B probe:

```bash
uv run scripts/pmpp-hard/validate-inputs.py --probe-outputs \
  --parameter 1 --parameter 1 --parameter 0xffffffffffffffff \
  --output /tmp/pmpp-scenario-output-probes v2-pl-30-graph_frontier_bfs_state
```

The probe hashes guarded output-buffer downloads, requires every oracle check to pass, and rejects differing outputs for a repeated parameter or unchanged outputs across distinct parameters. It requires explicit repeated and distinct parameters and is not supported for Triton. `--parameter` is a repeatable diagnostic-only override of `--rollouts`; production scoring still draws fresh parameters. Run ordinary uninstrumented fresh draws separately. Result records retain the correctness policy, parameter, reference hash and uninstrumented grader-source hashes. These probes establish variation/reproducibility, not immunity to precomputation.

## Performance inputs and paired-gate validation

```bash
uv run scripts/pmpp-hard/validate-benchmarks.py --paired --output /tmp/pmpp-bench-validation
uv run scripts/pmpp-hard/validate-negative-controls.py --output /tmp/pmpp-negative-validation
```

The first command enumerates all 48 effective performance tasks. It checks each seed consumer, runs the release reference on A/A/B inputs (A is `2**64-1` to exercise overflow), and optionally runs the real five-pair scorer on a fresh seed. Per-task logs, seed values, source/image identities, measurements and coverage are saved separately. Use positional task IDs for a subset and `--seed 0x...` to select A. Triton probes fingerprint samples of generated random tensors and require every observed oracle comparison to pass, because some tolerance-based output digests encode oracle agreement rather than output bytes; CUDA/CUTLASS checks use changed output digests. These are input-variation checks, not exhaustive generator or security proofs. Instrumented input-probe timings are not calibration data. The separate paired pass is uninstrumented, but still requires an idle host for calibration. A failed reference-versus-reference ratio is recorded as `paired_pass=false` even if the complete-pair/input checks pass.

The second command uses the actual paired scorer with nine deliberately bad submissions: CUDA task 01 and CUTLASS task 13's no-op starters (with a nonzero workspace request to reach output verification), and all seven active Triton references wrapped to reuse the first output for each argument shape while ignoring new tensor values. These controls intentionally bypass earlier correctness and source gates to isolate performance-output verification. A successful control must actually build and be rejected for a digest mismatch, not merely fail to compile or lose on timing. This is not a comprehensive offline-precomputation attack suite.

Both commands use only local GPU Docker and the existing images, and require an empty output directory. They do not launch models, modify release data or publish anything. Run GPU validation commands sequentially, without concurrent training or agent self-checks on the same device.

## Fresh diagnostics for saved submissions

```bash
uv run scripts/pmpp-hard/regrade-saved.py \
  --traces /absolute/path/to/historical/traces.jsonl \
  --ids v2-pl-18-triton_flash_decode v2-pl-55-lock_manager_deadlock v2-pl-66-mk_schedule_planner \
  --rollouts 2 --output /tmp/pmpp-saved-diagnostics
```

This reads captured submission sources, runs the current clean scorer with fresh correctness/performance inputs, and writes new diagnostic records with the original trace ID, source location/hash, current policies, images, metrics and logs. Historical traces are read-only; new metrics are not inserted into old rollouts or reported as a new model evaluation. Multiple trace files are supported; each matching historical attempt is replayed independently. Use `--residency-time-mode enforce` only for an explicitly labeled old-floor comparison. No task prompt, model, or harness is rerun.

To separate timing noise from performance-input variation, run repeated diagnostics on an idle GPU with `--benchmark-seed 0x...` using a seed from a private trace. This optional support-script override is marked in both the diagnostic manifest and each trace; it does not add a fixed-seed option to production tasksets. Correctness inputs remain fresh. These fixed-seed diagnostics must not be counted as independent model evaluations.
