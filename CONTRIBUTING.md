# Contributing

Residents contribute through pull requests from repository branches or forks.
`main` only moves through reviewed PRs.

1. Branch from `main`: `feat/<env-name>`, `fix/<...>`, or `chore/<...>`.
2. Add or update an environment under `environments/<env-name>/`.
3. Open a draft PR and keep pushing to it as you iterate.
4. A company code owner reviews the PR. After approval and passing checks, it
   can be squash-merged.

## CI

- Ruff and package checks run on every PR update, including drafts.
- Bugbot and Macroscope run automatically.

## Local checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -n auto tests -v
```

Every environment needs a `pyproject.toml` and `README.md` and must build with
`uv build`.
