# Prime Agent

PhySim uses the `prime-agent` harness included in its pinned Verifiers version.
There is no PhySim-specific harness to install or maintain. Keep the runtime
and bundle settings from `configs/physim/eval.toml`, replace the harness block,
and select the IPython wording for the public coding interface:

```toml
[env.agent.harness]
id = "prime-agent"

[env.taskset.task]
coding_interface = "ipython"
```

Remove Bash-only options such as `edit` and `search` when changing the block.
Verifiers installs the pinned Prime Agent during trusted setup and provides its
model interception, MCP connection, and framework-only network policy. Harness
provider dialects, caching, and supported model settings belong to Verifiers.

The published BF case study used the older `physim_prime_agent` adapter with
Verifiers 0.3.1 and Prime Agent 0.9.5. Its exact reproduction instructions remain
[in the study repository](https://github.com/swpo/physim/blob/main/environments/physim/PRIME_AGENT.md).
