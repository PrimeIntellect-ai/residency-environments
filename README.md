# residency-environments

RL environments built by the Prime Intellect **RL Residency**.

This repo mirrors the structure of `research-environments`: each environment
lives in `environments/<env_name>_v1/` as an installable Verifiers v1 taskset
run via the `eval` CLI.

## Setup

```bash
uv sync
uv run pre-commit install
```

## Repo layout

- `environments/` — one directory per environment (installable package)
- `tests/` — harness tests that install and smoke-eval every v1 taskset
- `scripts/` — setup helpers

## Contributing

The repo is private until public release of the work in it. Residents have
write access and contribute via branch → PR; `main` only moves through
reviewed PRs. See [CONTRIBUTING.md](CONTRIBUTING.md).
