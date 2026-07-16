## What

<!-- Which environment does this add or change, and why? -->

## How to try it

<!-- Include the taskset ID, runtime requirements, and a short eval command. -->

## Data and scoring

<!-- Dataset source/license, reward design, and known reward-hacking risks. -->

## Checklist

- [ ] `uv run ruff check . && uv run ruff format --check .` passes
- [ ] `uv run pytest -n auto tests -m "not integration" -v` passes locally
- [ ] Environment has a `pyproject.toml` and `README.md`
