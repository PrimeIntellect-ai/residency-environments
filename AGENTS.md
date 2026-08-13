# AGENTS.md

## Writing code

- **Minimal try/except**: let errors propagate — silent failures hide bugs. Only catch exceptions for intentional fault tolerance (retries, robustness).
- **Targeted comments**: don't explain your work process or reference old code. Use targeted comments sparingly to clarify ambiguous logic.
- **Zen of Python**: remember the Zen of Python when writing code.
```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

## Running code

- **Always use uv**: run code with `uv run` or `uv run <command>`, never raw `python`.
- **Adding dependencies**: add to `pyproject.toml` and run `uv sync --all-extras` to install and lock them.
- **Git dependency pins**: when pinning git dependencies in `pyproject.toml`, always use a small (7-char) commit hash for the `rev` field.

## Testing

Write tests as plain functions with pytest fixtures. Don't use class-based tests.

## Git

- **Branch prefixes**: use the following prefixes for branches: `feat/`, `fix/`, `chore/`

## GitHub

- **Draft PRs**: always create PRs as drafts (`gh pr create --draft`) so work-in-progress is clear; CI still runs on draft updates.
- **Pull requests**: do not include a "test plan" section in PR descriptions unless you actually ran tests to verify the changes or the user explicitly asked for one.

## Repository Development Notes

Use this guidance when contributing to the `residency-environments` repository itself.

- Always use `uv` to run Python commands
- During development, install environments (`/environments`) from the project's root directory using editable, local installs as `uv pip install -e ./environments/<env-name>`. DO NOT install from within the environment directories.
- Every directory under `environments/` is a Verifiers v1 environment; no naming suffix is required.
- For synthetically generated environments, the data generation and validation code lives in a matching `generators/<env-name>/` directory (see `generators/README.md`).
- To check an environment implementation, use the v1 `eval` CLI. Start with a single example and two rollouts so tasksets using group rewards can compare outputs.
```bash
uv run eval --taskset.id <env-name> -n 1 -r 2 --max-turns 4
```
- After comprehensive changes, check linting and styling for the environment you modified
```bash
uv run ruff check ./environments/<env-name>
uv run ruff format --check ./environments/<env-name>
```
- Always keep the environment's README up-to-date with any relevant changes.
