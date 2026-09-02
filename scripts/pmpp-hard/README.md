# PMPP Hard runtime images

Build the three local GPU images from the repository root:

```bash
scripts/pmpp-hard/build-images.sh
```

The CUDA agent image supplies CUDA 12.8, Python, and the utilities needed by a coding harness. All three images pin `uv` and pre-cache the dependencies used to launch the configured Verifiers harness with agent network access disabled. Triton tasks use a CUDA 12.8 image with pinned PyTorch and Triton versions. The CUTLASS image includes the pinned CUTLASS headers used by the release task.

All images require an NVIDIA Container Toolkit installation and a compatible host driver. The image definitions pin their base image and specialized toolchain inputs. Publish equivalent images to an accessible registry before using a remote GPU runtime.
