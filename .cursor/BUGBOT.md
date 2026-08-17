# Bugbot Instructions

## Eval Convention Enforcement

Every environment package must declare its full-eval convention in its `pyproject.toml`:

```toml
[tool.verifiers.eval]
num_examples = 50          # how many tasks a representative eval uses
rollouts_per_example = 8   # rollouts per task
```

When reviewing changes under `environments/`:

1. **Require the table on new environments**: if a PR adds an environment whose `pyproject.toml` has no `[tool.verifiers.eval]` table, request one.
2. **Check it stays sensible**: if a PR changes the taskset size or difficulty materially, ask whether `num_examples` still gives a representative eval.
