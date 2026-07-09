# residency-environments

RL environments built by the Prime Intellect **RL Residency**.

This repo mirrors the structure of `research-environments`: each environment
lives in `environments/<env_name>/` as an installable package. Classic (v0)
verifiers environments expose `load_environment(...)`; `*_v1` packages are
verifiers-v1 tasksets run via the `eval` CLI.

## Setup

```bash
uv sync
uv run pre-commit install
```

## Repo layout

- `environments/` — one directory per environment (installable package)
- `tests/` — harness tests that install, import, load, and smoke-eval every env
- `scripts/` — setup helpers

## Contributing

The repo is private until public release of the work in it. Residents have
write access and contribute via branch → PR; `main` only moves through
reviewed PRs. See [CONTRIBUTING.md](CONTRIBUTING.md).
