# AGENTS.md

## Writing code

- **Minimal try/except**: let errors propagate — silent failures hide bugs. Only catch exceptions for intentional fault tolerance (retries, robustness).
- **Targeted comments**: don't explain your work process or reference old code. Use targeted comments sparingly to clarify ambiguous logic.
- **Zen of Python**: remember the Zen of Python when writing code.

```text
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

Don't add per-environment unit tests; environments are validated by actually running them (`uv run eval ...`). Repo-level package checks live in `tests/` and should not be modified.

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
- Every environment must declare its full-eval convention in `pyproject.toml` under `[tool.verifiers.eval]` (`num_examples`, `rollouts_per_example`); automated eval runs read it when no sample count is given. Infinite tasksets fall back to 50 examples.
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

## Environment Design

Always base your environment on the newest version of verifiers v1.

### Structure

Tasksets go in *environments*.

Questions/task descriptions, and possibly gold standard answers go into a HuggingFace dataset. If there's large amounts of data that the model has to work with, which takes time to setup in a sandbox, ship it in sandbox images and reference them in the Huggingface dataset for every prompt, so that each task starts with the right sandbox.

If the data is generated synthetically, the scripts belong under *generators*.

The structure is always:

- environments
  - `<env-name>/<code>`
- generators
  - `<env-name>/<code>`  # if data is synthetically generated

### On tools

You generate a taskset, not a harness. Most environments need to offer zero tools to the models, that's an outdated mode of operation. Nowadays, LLMs run in coding harnesses, which offer extremely general tools that enable working on effectively any task, for any model. There are some situations where special tools are required, but those are very rare and if you find yourself implementing a tool for use by the agent, double-check if this couldn't be done by a plain harness instead. If yes, it should be so, because that enables a big axis of diversification of the environment: solving it with different coding harnesses, which verifiers enables out of the box.

You should rather think about whether your model requires containerization, if it is allowed to have internet access, etc.

### Prompting

Don't use role prompts.

> Reasoning: Pretraining makes LLMs simulators. Posttraining should turn them into consistent agents. The mechanism for this is often a shortcut, where the model learns to simulate an agent given a certain prompt, but not internalizing the agentic-ness fully. Role prompts during RL bind agentic behavior to certain prompts, which makes prompt injection attacks easier and performance more spiky.

Keep prompts minimal without making them too short.

> Reasoning: During RL training, prompts should only give the model the information it needs to know what the task is and what tools it has available. The strategy for solving the task should be found out through RL training pressure, not prescribed. This is important because RL training will potentially lead to strategies that the prompter didn't consider, making restrictive prompts an unnecessary constraint on final behavior. Secondly, if RL pressure still overcomes that constraint and improves the model beyond it, the prompt now contradicts the behavior, which means that we have trained against instruction following.

Instructions and reward must match.

> Reasoning: any contradiction between the instructions and reward will teach the model anti-instruction following. RL will teach the model to ignore the prompt and take the route that the reward points toward. This is very bad.

Don't tell the model about behaviors that should generalize, tell it about ones that are specific to the environment.

> Reasoning: if the model is told to behave in some way, and rewarded accordingly, it will learn to follow the instruction and behave that way given the instruction. If it isn't told this, but is still rewarded for the behavior, it will make the behavior a default that has to actively overridden. Example: good, consice writing style must generalize -> don't explicitly prompt the model for it, just apply the reward. Don't look at some webpage because in the given environment it would make the task trivial -> this shouldn't generalize, we want the model to be free to look at anything; we just want to make the task harder in this environment. Side effect: adding the instruction and fitting reward will teach instruction following.

Tool descriptions are prompts; apply any rules that make sense for them.

> Reasoning: they're added to the system prompt.

Tool descriptions are prompts belonging to the tool; describe what the tool does, not how to use it in your environment.

> Reasoning: tool use should generalize across domains. The model should be able to figure out how to use a tool for a given task without being told so. This is analogous to the taskset <-> harness duality in verifiers: a harness is very general and can apply to any taskset. Custom tools should be prompted with this in mind.
