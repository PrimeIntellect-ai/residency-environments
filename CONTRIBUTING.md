# Contributing

This repo hosts RL environments built by the Prime Intellect RL Residency.
It is private: work stays here until its public release (blog post, publishing
the environment to the Environments Hub, etc.).

## Workflow

You have write access: push branches to this repo directly — no fork needed.
Nobody can push to `main` (not even admins); everything lands via PR.

1. **Branch** off `main`: `feat/<env-name>`, `fix/<...>`, or `chore/<...>`.
2. **Develop** your environment under `environments/<env_name>/` (see
   `AGENTS.md` for conventions and how to run things locally).
3. **Open a PR** against `main`. Style checks and unit tests run automatically.
4. A maintainer reviews your PR. Merging requires an approving review from a
   code owner (a Prime Intellect maintainer) — other approvals are welcome as
   informal review but don't unlock the merge.

## CI

- **Style** (ruff) and **unit tests** run on every PR. They run without any
  secrets, so anything requiring API keys is skipped automatically — make sure
  your env's install/import/load paths work without credentials.
- **Integration tests** (real model rollouts, needs API keys) run when a
  maintainer adds the `safe-to-test` label, and each run additionally requires
  a maintainer's approval before secrets are released (remove + re-add the
  label to re-run after new commits).
- Code-review bots (Bugbot, Macroscope) are triggered manually by maintainers
  only.

## Before opening a PR

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -n auto tests -v
```

Every environment needs a `pyproject.toml` with real name/version/description/
keywords and a `README.md` — the unit tests enforce this.
