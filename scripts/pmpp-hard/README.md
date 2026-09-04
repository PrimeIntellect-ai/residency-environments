# PMPP Hard runtime images

Build the three local GPU images from the repository root:

```bash
scripts/pmpp-hard/build-images.sh
```

The CUDA agent image supplies CUDA 12.8, Python, and the utilities needed by a coding harness. All three images pin `uv` and pre-cache the dependencies used to launch the configured Verifiers harness with agent network access disabled. Triton tasks use a CUDA 12.8 image with pinned PyTorch and Triton versions. The CUTLASS image includes the pinned CUTLASS headers used by the release task.

All images require an NVIDIA Container Toolkit installation and a compatible host driver. The image definitions pin their base image and specialized toolchain inputs. Publish equivalent images to an accessible registry before using a remote GPU runtime.

To check all 45 randomized correctness graders and task 09's independently reseeded sanity grader against their trusted release references, run:

```bash
uv run scripts/pmpp-hard/validate-inputs.py --output /tmp/pmpp-input-validation
```

This runs two independent input draws per randomized task and two checks of the fixed task 09 sanity cases on local GPU Docker, with external network access blocked. It writes per-run logs and a JSON result manifest to the chosen output directory. It checks correctness without model calls or performance measurements; full evaluation still uses the configs under `configs/pmpp-hard/`.
