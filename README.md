# residency-environments

RL environments built by the Prime Intellect **RL Residency**.

Each directory in `environments/` is an installable Verifiers v1 taskset.
Supporting data generators, development scripts, and reusable test configurations
live in `generators/`, `scripts/`, and `configs/`, respectively.

## Setup

```bash
uv sync
uv run pre-commit install
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.
