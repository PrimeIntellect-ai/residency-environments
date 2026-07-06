# Contributing

This repo hosts RL environments built by the Prime Intellect RL Residency.
It is public, but interaction is limited to collaborators: only residents
(added as collaborators) and Prime Intellect maintainers can open issues and
pull requests.

## Workflow for residents

You have triage access: you can open issues and PRs, but you cannot push
branches to this repo or merge. The flow is:

1. **Fork** the repo on GitHub.
2. **Branch** in your fork: `feat/<env-name>`, `fix/<...>`, or `chore/<...>`.
3. **Develop** your environment under `environments/<env_name>/` (see
   `AGENTS.md` for conventions and how to run things locally).
4. **Open a PR** against `main`. Style checks and unit tests run automatically.
5. A maintainer reviews your PR. Merging requires an approving review from a
   code owner (a Prime Intellect maintainer) — resident approvals are welcome
   as informal review but don't unlock the merge.

## CI

- **Style** (ruff) and **unit tests** run on every PR. They run without any
  secrets, so anything requiring API keys is skipped automatically — make sure
  your env's install/import/load paths work without credentials.
- **Integration tests** (real model rollouts, needs API keys) only run after a
  maintainer adds the `safe-to-test` label. The label is removed automatically
  when you push new commits, so a maintainer re-approves each iteration.
- Code-review bots (Bugbot, Macroscope) are triggered manually by maintainers
  only.

## Before opening a PR

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -n auto tests -v
```

Every environment needs a `pyproject.toml` with real name/version/description/
keywords and a `README.md` — the unit tests enforce this.
