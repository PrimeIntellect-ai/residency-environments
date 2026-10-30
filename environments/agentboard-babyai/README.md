# AgentBoard BabyAI

A self-contained, text-based BabyAI environment for navigation and object manipulation, adapted from AgentBoard for Verifiers.

Requires Python 3.11+ and Verifiers 0.4.x. The runtime, prompt, and 112 task annotations are bundled; an AgentBoard checkout is not required. The split is 84 training episodes and 28 held-out episodes across 28 task families.

## Evaluation

From this repository's root, install the package and use the supplied Prime Inference configuration. Authenticate first with `prime login` or `PRIME_API_KEY`.

```bash
uv pip install -e ./environments/agentboard-babyai
uv run --no-sync vf-eval @ configs/agentboard-babyai/eval.toml -n 1 -r 2
```

Omit `-n 1 -r 2` to evaluate all 28 held-out episodes once. The configuration uses Qwen3.5-9B with thinking disabled, a 20-turn limit, and the tool-free `null` harness.

## Licensing

The AgentBoard runtime and prompt are under Apache 2.0; the task annotations are under GPL 2.0. See `NOTICE.md`, `UPSTREAM_AGENTBOARD.md`, and `licenses/` for attribution and license details.
