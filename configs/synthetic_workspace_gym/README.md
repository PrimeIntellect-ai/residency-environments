# Synthetic Workspace Gym configurations

The frozen V5 curricula live in the pinned Hugging Face dataset, not in this repository. `release.json` records the dataset commit, export integrity hash, exporting generator commit, and SHA-256 hashes of the six source manifests. Earlier published dataset revisions remain frozen on Hugging Face.

To reproduce the source manifests without adding generated data to Git, run `scripts/synthetic_workspace_gym/build_qwen35_training_manifests.py`. It writes to the ignored `generators/synthetic_workspace_gym/build/manifests/` directory by default. Compare those files with the hashes in `release.json` before attempting to reproduce the pinned export. The installable taskset loads the published curriculum rows from the pinned dataset commit.
