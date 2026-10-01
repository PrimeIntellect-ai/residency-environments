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
The laboratory contract refers to operations; Prime Agent discovers their
actual MCP names or wrappers through its own tool interface.
Verifiers installs the pinned Prime Agent during trusted setup and provides its
model interception, MCP connection, and framework-only network policy. Harness
provider dialects, caching, and supported model settings belong to Verifiers.

For hosted Qwen3.5-9B, use `configs/physim/qwen-prime-agent.toml`. It sets the
native model context window to 65,536 tokens and enables earlier compaction
with room for the configured response size. `qwen-pi.toml` provides the same
context settings for Pi's Chat Completions transport. These are provider-specific
settings, not model-family defaults. The adapters pass `sampling.max_tokens`
through to the native model's output limit as well.

Successful submission freezes the artifact and closes experimentation; the
harness returns its final answer naturally. Inspect `artifact_source` and
`submitted_before_finalization` in the limit audit to distinguish submission
from automatic collection after a stop or limit.

The published BF case study used the older `physim_prime_agent` adapter with
Verifiers 0.3.1 and Prime Agent 0.9.5. Its exact reproduction instructions remain
[in the study repository](https://github.com/swpo/physim/blob/main/environments/physim/PRIME_AGENT.md).
