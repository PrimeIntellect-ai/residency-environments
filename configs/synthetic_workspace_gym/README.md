# Synthetic Workspace Gym configurations

These immutable manifests define the task assignments used for SFT, RL, validation, and difficulty calibration. `scripts/synthetic_workspace_gym/build_qwen35_training_manifests.py` rebuilds them deterministically, and the generator package consumes them when exporting frozen task data for the installable environment.

The installable taskset keeps the manifest selections it needs as package data so its wheel remains self-contained. These files are the repository-level source configurations used to reproduce that data.
